def main():
    from marathon.io import read_yaml

    from lorem import comms

    settings = read_yaml("settings.yaml")

    # -- settings --

    from marathon.data import datasets

    data_train = datasets / settings.pop("train")
    data_valid = datasets / settings.pop("valid")

    # {name: (source, save_predictions)}
    test_datasets = {
        "valid": (data_valid, False),
    }
    if "test_datasets" in settings:
        for k, (data, save) in settings.pop("test_datasets").items():
            test_datasets[k] = (datasets / data, save)

    batcher_config = settings.pop("batcher")
    if "kind" in batcher_config:
        import importlib

        handle = batcher_config.pop("kind")
        batcher_class = getattr(
            importlib.import_module(".".join(handle.split(".")[:-1])),
            handle.split(".")[-1],
        )
    else:
        batcher_class = None

    should_filter_mixedpbc = settings.pop("filter_mixed_pbc", False)
    should_filter_above_num_atoms = settings.pop("filter_above_num_atoms", False)

    loss_weights = settings.pop("loss_weights", {"energy": 0.5, "forces": 0.5})
    scale_by_variance = settings.pop("scale_by_variance", False)

    use_rotational_augmentation = settings.pop("rotational_augmentation", False)

    start_learning_rate = float(settings.pop("start_learning_rate", 1e-3))
    min_learning_rate = float(settings.pop("min_learning_rate", 1e-6))

    max_epochs = settings.pop("max_epochs", 2000)
    valid_every_epoch = settings.pop("valid_every_epoch", 2)

    optimizer_name = settings.pop("optimizer", "adam")

    # lr decay
    decay_style = settings.pop("decay_style", "linear")
    start_decay_after = settings.pop("start_decay_after", 10)
    stop_decay_after = settings.pop(
        "stop_decay_after", max_epochs
    )  # ignored for exponential
    warmup_epochs = settings.pop("warmup_epochs", 0)
    if warmup_epochs and decay_style != "warmup_cosine":
        comms.warn(f"requested warmup, but schedule {decay_style} has no warmup!")

    # gradient clipping
    gradient_clip = settings.pop("gradient_clip", 0)

    which_checkpointers = settings.pop("checkpointers", "default")

    seed = settings.pop("seed", 0)
    print_model_summary = True
    benchmark_pipeline = settings.pop("benchmark_pipeline", True)
    workdir = "run"

    use_wandb = settings.pop("use_wandb", True)
    # used for wandb -- use folder names by default
    wandb_project = None
    wandb_name = None

    default_matmul_precision = settings.pop("default_matmul_precision", "default")
    debug_nans = settings.pop("debug_nans", False)  # ~50% slowdown, use with care
    enable_x64 = settings.pop("enable_x64", False)

    compilation_cache = settings.pop("compilation_cache", False)

    # after this many compiles of do_batch, we clear the cache to save RAM
    compilation_cache_max_size = settings.pop("compilation_cache_size", 50)

    # loss mean() is weighted by actual samples
    correct_mean = settings.pop("correct_mean", True)

    # settings for grain
    worker_count = settings.pop("worker_count", 4)
    worker_count_valid = settings.pop("worker_count_valid", worker_count)
    worker_buffer_size = settings.pop("worker_buffer_size", 2)
    worker_buffer_size_valid = settings.pop("worker_buffer_size_valid", worker_buffer_size)

    # if we didn't use all the settings, emit a warning
    if len(settings) > 0:
        comms.warn("didn't use all entries in settings.yaml, we ignored:")
        comms.warn(f"... {', '.join(settings.keys())}", full=True)

    # -- imports & startup --

    import numpy as np
    import jax
    import jax.numpy as jnp

    from pathlib import Path

    jax.config.update("jax_enable_x64", enable_x64)
    jax.config.update("jax_default_matmul_precision", default_matmul_precision)
    jax.config.update("jax_debug_nans", debug_nans)

    if compilation_cache:
        if isinstance(compilation_cache, str):
            cache_dir = Path(compilation_cache)
        else:
            cache_dir = Path("../.jax_cache/")

        comms.talk(
            f"caching compilations to {cache_dir.resolve()}",
            full=True,
        )
        cache_dir.mkdir(exist_ok=True)

        # see: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
        jax.config.update("jax_compilation_cache_dir", str(cache_dir.resolve()))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 10.0)
        # this somehow breaks everything on kuma as of january 2026
        # jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

    reporter = comms.reporter()
    reporter.start("run")
    reporter.step("startup")

    # -- housekeeping based on settings --
    keys = list(loss_weights.keys())
    use_stress = "stress" in keys
    batcher_config["keys"] = keys

    workdir = Path(workdir)

    # -- randomness --
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)

    # -- model --
    from marathon.io import from_dict

    model_config = read_yaml("model.yaml")
    assert "model" in model_config
    if "baseline" in model_config:
        species_to_weight = model_config["baseline"]["elemental"]
    else:
        species_to_weight = None

    model = from_dict(model_config["model"])
    cutoff = model.cutoff

    params = model.init(init_key, *model.dummy_inputs())

    if print_model_summary:
        from flax import linen as nn

        msg = nn.tabulate(model, init_key)(*model.dummy_inputs())
        comms.state(msg.split("\n"), title="Model Summary")

    num_parameters = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    comms.state(f"Parameter count: {num_parameters}")

    # -- checkpointers --
    from marathon.emit import SummedMetric

    checkpointers = []

    name = "R2_" + "+".join([k[0].upper() for k in keys])
    checkpointers.append(SummedMetric(name, "r2", keys=keys))

    # per-key MAE checkpointers
    for k in keys:
        name = "MAE_" + k[0].upper()
        checkpointers.append(SummedMetric(name, "mae", keys=[k]))

    if which_checkpointers == "full":
        for k in keys:
            name = "RMSE_" + k[0].upper()
            checkpointers.append(SummedMetric(name, "rmse", keys=[k]))

    checkpointers = tuple(checkpointers)

    # -- data loading --
    from marathon.evaluate.metrics import get_stats
    from marathon.extra.hermes import (
        DataLoader,
        DataSource,
        FilterAboveNumAtoms,
        FilterEmpty,
        FilterMixedPBC,
        FilterNoop,
        IndexSampler,
        RandomRotation,
        prefetch_to_device,
    )
    from marathon.extra.hermes.pain import Record, RecordMetadata
    from marathon.utils import tree_stack

    def get_batcher(valid=False):
        conf = batcher_config
        if valid:
            conf["drop_remainder"] = False

        if not batcher_class:
            return model.to_batch(**conf)
        return batcher_class(**conf)

    batcher = get_batcher()  # only used for its class methods

    # -- setup filters for dataloaders --
    filter_empty = FilterEmpty()

    filters = [FilterNoop()]
    if should_filter_mixedpbc:
        filters.append(FilterMixedPBC())
    if should_filter_above_num_atoms:
        threshold = should_filter_above_num_atoms
        filters.append(FilterAboveNumAtoms(threshold))

    # -- data sources --
    source_train = DataSource(data_train, species_to_weight=species_to_weight)
    if species_to_weight is None:
        species_to_weight = source_train.species_to_weight
        comms.talk("using baseline from training data")
    source_valid = DataSource(data_valid, species_to_weight=species_to_weight)
    baseline = {"elemental": species_to_weight}

    properties = source_train.properties
    batcher_config["properties"] = properties
    comms.talk(f"properties: {list(properties.keys())}")

    to_sample = model.to_sample(cutoff=cutoff, keys=keys, properties=properties)

    n_train = len(source_train)
    n_valid = len(source_valid)

    max_steps = max_epochs * n_train
    valid_every = valid_every_epoch * n_train
    comms.talk(f"run for {max_epochs} epochs, {max_steps} steps", full=True)
    comms.talk(
        f"validate every {valid_every_epoch} epochs, every {valid_every} steps",
        full=True,
    )

    # -- setup for the validation dataloader --
    reporter.step("loading validation set")

    if "batch_size" in batcher_config:
        # avoid edge case: pipeline not return anything
        #                  if batch_size * workers < n_valid
        min_worker_count = n_valid // batcher_config["batch_size"]
        if min_worker_count < worker_count_valid:
            worker_count_valid = min_worker_count
            comms.talk(f"worker_count_valid set to {worker_count_valid}")

    def get_valid_iterator():
        return iter(
            DataLoader(
                data_source=source_valid,
                sampler=IndexSampler(
                    n_valid,
                    num_epochs=1,
                    seed=seed,
                    shuffle=False,
                ),
                operations=[
                    *filters,
                    to_sample,
                    FilterEmpty(),
                    get_batcher(valid=True),
                ],
                worker_count=worker_count_valid,
                worker_buffer_size=worker_buffer_size_valid,
            )
        )

    # -- obtaining statistics for validation set --
    def valid_samples():
        for i in range(n_valid):
            atoms = source_valid[i]
            if all([f.filter(atoms) for f in filters]):
                sample = to_sample.map(atoms)
                if filter_empty.filter(sample):
                    yield sample

    valid_stats = get_stats(valid_samples(), keys=keys, properties=properties)

    valid_batch_sizes = np.array(
        [batcher.count_samples(batch) for batch in get_valid_iterator()]
    )
    median_valid_batch_size = int(np.median(valid_batch_sizes))

    if scale_by_variance:
        old_loss_weights = loss_weights

        loss_weights = {k: v / valid_stats[k]["var"] for k, v in loss_weights.items()}

        msg = []
        for k, v in loss_weights.items():
            msg.append(f"{k}: {old_loss_weights[k]:.3f} -> {v:.3f}")
        comms.state(msg, title="variance scaled loss weights")

    reporter.step("setup training pipeline")

    def get_training_iterator(num_epochs):
        if use_rotational_augmentation:
            operations = [
                RandomRotation(),
                *filters,
                to_sample,
                FilterEmpty(),
                get_batcher(),
            ]
        else:
            operations = [
                *filters,
                to_sample,
                FilterEmpty(),
                get_batcher(),
            ]

        return iter(
            DataLoader(
                data_source=source_train,
                sampler=IndexSampler(
                    n_train,
                    num_epochs=num_epochs,
                    seed=seed,
                ),
                operations=operations,
                worker_count=worker_count,
                worker_buffer_size=worker_buffer_size,
            )
        )

    if benchmark_pipeline:
        from time import monotonic

        probe_epochs = 2

        reporter.step(
            f"benchmark pipeline ({probe_epochs} epochs)",
            spin=False,
        )

        @jax.jit
        def test_fn(batch):
            return batcher.info(batch)

        # trigger jit (the first fews times...)
        it = get_training_iterator(probe_epochs)
        i = 0
        for batch in it:
            test_fn(batch)
            i += 1

        compiles = test_fn._cache_size()
        comms.talk(
            f"{compiles} compilations in {probe_epochs} epochs ({compiles / (i + 1):.2f}/batch)"
        )

        test_iter = prefetch_to_device(get_training_iterator(probe_epochs), 2)

        results = []
        start = monotonic()
        for i, batch in enumerate(test_iter):
            reporter.tick(f"batch {i}")
            results.append(test_fn(batch))
            del batch
        duration = monotonic() - start

        jax.clear_caches()

        results = tree_stack(results)
        results = jax.tree.map(lambda x: np.array(x, dtype=np.int64), results)
        real, total, shape = results

        real_samples = real["samples"].sum()
        pipeline_speed = duration / real_samples
        num_batches = i + 1

        utilization = {}
        shapes = {}
        for k in real.keys():
            util = (real[k] / total[k]).mean()
            utilization[k] = 100.0 * util

        for k, v in shape.items():
            shapes[k] = np.unique(v)

        all_shapes = np.array(list(shape.values()))

        # find the top 5 most common input shapes
        top_n = 5

        def unique_outcomes(X):
            U, inv, c = np.unique(X, axis=1, return_inverse=True, return_counts=True)

            o = np.argsort(c)[::-1]

            return U[:, o][:, :top_n], c[o][:top_n]

        top_shapes, counts = unique_outcomes(all_shapes)
        top_shapes = np.round(np.log2(top_shapes), 2)

        msg = []
        msg.append(
            f"incidence: {' '.join(map(lambda x: f'{100 * x:.2f}%', counts / num_batches))}"
        )
        for i, k in enumerate(shape.keys()):
            msg.append(f"{k:<10}: {str(top_shapes[i])}")

        comms.state(msg, title=f"Top {top_n} shapes (log2)")

        msg = []
        msg.append(f"speed       : {1e6 * pipeline_speed:.0f}\u00b5s/sample")
        msg.append(
            f"              {worker_count} train workers, buffer {worker_buffer_size}"
        )
        msg.append(
            f".             {worker_count_valid} valid workers, buffer {worker_buffer_size_valid}"
        )

        msg.append("")
        msg.append("utilization (real/padded) (average over batches):")
        for name, value in utilization.items():
            msg.append(f"{name:<10}: {value:.2f}%")

        msg.append("")
        msg.append("unique sizes (reported as log2):")
        for name, value in shapes.items():
            msg.append(f"{name:<10}: {np.round(np.log2(value), 1)}")

        m = "-> number of unique sizes: "
        m += ", ".join([f"{len(value)} {name.strip()}" for name, value in shapes.items()])
        msg.append(m)

        msg.append("")
        msg.append(
            f"num batches: {num_batches} ({real_samples / num_batches:.0f} samples/batch)"
        )

        comms.state(msg, title="Training Pipeline Statistics")

        for name, value in utilization.items():
            if value < 50:
                comms.warn(f"utilization (real/padded) of {name} is {value:.1f}%<50%!")

        if compiles > 100:
            comms.warn(f"expecting {compiles}>100 compilations")

        median_train_batch_size = int(np.median(real["samples"]))

        median_batch_size = median_train_batch_size
        batches_per_epoch = num_batches
    else:
        pipeline_speed = 0.0
        median_batch_size = median_valid_batch_size
        batches_per_epoch = int(len(source_train) / median_batch_size)

    comms.talk(f"estimated samples/batch: {median_batch_size}")
    comms.talk(f"estimated batches/epoch: {batches_per_epoch}")

    iter_train = get_training_iterator(max_epochs)

    # -- optimizer --
    import optax

    reporter.step("setup optimizer")

    if decay_style == "linear":
        transition_steps = stop_decay_after * batches_per_epoch
        initial_steps = start_decay_after * batches_per_epoch
        scheduler = optax.schedules.linear_schedule(
            init_value=start_learning_rate,
            end_value=min_learning_rate,
            transition_begin=initial_steps,
            transition_steps=transition_steps - initial_steps,
        )

    elif decay_style == "exponential":
        transition_steps = max_epochs * batches_per_epoch
        initial_steps = start_decay_after * batches_per_epoch
        decay_rate = min_learning_rate / start_learning_rate
        scheduler = optax.schedules.exponential_decay(
            init_value=start_learning_rate,
            transition_steps=transition_steps - initial_steps,
            transition_begin=initial_steps,
            decay_rate=decay_rate,
            end_value=min_learning_rate,
        )
    elif decay_style == "warmup_cosine":
        scheduler = optax.schedules.warmup_cosine_decay_schedule(
            init_value=min_learning_rate,
            peak_value=start_learning_rate,
            warmup_steps=warmup_epochs * batches_per_epoch,
            decay_steps=max_epochs * batches_per_epoch,
            end_value=min_learning_rate,
        )

    if optimizer_name == "muon":
        opt = optax.contrib.muon
    else:
        opt = getattr(optax, optimizer_name)

    @optax.inject_hyperparams
    def optimizer(learning_rate):
        if gradient_clip:
            return optax.chain(optax.clip(gradient_clip), opt(learning_rate))
        return opt(learning_rate)

    optimizer = optimizer(scheduler)

    initial_opt_state = optimizer.init(params)

    # -- assemble state / handle restore --

    state = {
        "step": 0,
        "checkpointers": checkpointers,
        "opt_state": initial_opt_state,
        "iter_train": iter_train.get_state(),
    }

    if workdir.is_dir():
        from marathon.emit import get_latest

        comms.warn(
            f"found working directory {workdir}, will restore (only) model and optimisation state!"
        )
        reporter.step("restoring")

        items = get_latest(workdir, state)

        if items is None:
            comms.warn(f"failed to find checkpoints in workdir {workdir}, ignoring")
        else:
            params, state, new_model, _, _, _ = items

            comms.talk(f"restored step {state['step']}")

            # try to catch the most obvious error: editing the model config between restarts
            from marathon.io import to_dict

            assert to_dict(new_model) == to_dict(model)

            iter_train.set_state(state["iter_train"])
    else:
        workdir.mkdir()

    opt_state = state["opt_state"]

    # -- loggers --
    from marathon.emit import Txt
    from marathon.io import to_dict

    infra_info = {"package": "lorem"}

    reporter.step("setup loggers")

    training_pipeline = {
        "batcher": batcher_config,
        "use_rotational_augmentation": use_rotational_augmentation,
        "filter_mixed_pbc": should_filter_mixedpbc,
        "filter_above_num_atoms": should_filter_above_num_atoms,
    }

    if decay_style == "linear":
        lr_decay = {
            "style": "linear",
            "start_decay_after": start_decay_after,
            "stop_decay_after": stop_decay_after,
        }
    elif decay_style == "exponential":
        lr_decay = {
            "style": "exponential",
            "start_decay_after": start_decay_after,
        }
    elif decay_style == "warmup_cosine":
        lr_decay = {
            "style": "warmup_cosine",
            "warmup_epochs": warmup_epochs,
        }
    else:
        raise ValueError

    config = {
        "infra": infra_info,
        "n_train": n_train,
        "n_valid": n_valid,
        "loss_weights": loss_weights,
        "max_steps": max_steps,
        "max_epochs": max_epochs,
        "start_learning_rate": start_learning_rate,
        "min_learning_rate": min_learning_rate,
        "lr_decay": lr_decay,
        "optimizer": optimizer_name,
        "gradient_clip": gradient_clip,
        "training_pipeline": training_pipeline,
        "valid_every": valid_every,
        "model": to_dict(model),
        "num_parameters": num_parameters,
        "worker_count": worker_count,
        "worker_buffer_size": worker_buffer_size,
    }

    metrics = {key: ["r2", "mae", "rmse"] for key in keys}

    loggers = [Txt(metrics=metrics)]

    if use_wandb:
        import wandb
        from marathon.emit import WandB

        this_folder = Path.cwd()

        if wandb_project is None:
            wandb_project = f"{this_folder.parent.parent.stem}.{this_folder.parent.stem}"

        if wandb_name is None:
            wandb_name = f"{this_folder.stem}"

        run = wandb.init(config=config, name=wandb_name, project=wandb_project)

        config["wandb_id"] = run.id

        loggers.append(WandB(run, metrics=metrics))

    # -- setup actual training loop --
    from time import monotonic

    from marathon.emit import save_checkpoints
    from marathon.evaluate import (
        get_loss_fn,
        get_metrics_fn,
        get_predict_fn,
    )
    from marathon.utils import seconds_to_string as s2s

    reporter.step("setup training loop")

    if hasattr(model, "predict"):
        pred_fn = lambda params, batch: model.predict(params, batch, stress=use_stress)
    elif hasattr(model, "energy"):
        pred_fn = get_predict_fn(energy_fn=model.energy, stress=use_stress)
    else:
        pred_fn = get_predict_fn(apply_fn=model.apply, stress=use_stress)

    loss_fn = get_loss_fn(pred_fn, weights=loss_weights, correct_mean=correct_mean)
    loss_fn = jax.jit(loss_fn)

    train_metrics_fn = get_metrics_fn(keys=keys, properties=properties)
    valid_metrics_fn = get_metrics_fn(keys=keys, stats=valid_stats, properties=properties)

    # ... manager preamble

    def get_lr(opt_state):
        return float(opt_state.hyperparams["learning_rate"])

    def report_on_lr(opt_state):
        lr = get_lr(opt_state)
        return f"LR: {lr:.3e}"

    from marathon.emit.pretty import format_metrics

    class Manager:
        def __init__(
            self,
            state,
            interval,
            loggers,
            workdir,
            model,
            baseline,
            max_steps,
        ):
            self.state = state
            self.interval = interval
            self.loggers = loggers
            self.workdir = workdir
            self.model = model
            self.baseline = baseline

            self.max_steps = max_steps

            self.start_step = state["step"]
            self.start_time = monotonic()

            self.cancel = False

            self.compilations = 0

        @property
        def done(self):
            return self.step >= self.max_steps or self.cancel

        @property
        def step(self):
            return self.state["step"]

        @property
        def elapsed(self):
            return monotonic() - self.start_time

        @property
        def time_per_step(self):
            return self.elapsed / (self.step - self.start_step)

        @property
        def compute_time_per_step(self):
            return self.time_per_step - pipeline_speed

        @property
        def eta(self):
            return (self.max_steps - self.step) * self.time_per_step

        def should_validate(self, step):
            return step >= self.step + self.interval

        def report(
            self,
            step,
            params,
            opt_state,
            train_state,
            train_loss,
            train_metrics,
            valid_loss,
            valid_metrics,
            info={},
        ):
            assert step > self.step  # always forward

            self.state["step"] = step
            self.state["opt_state"] = opt_state
            self.state["iter_train"] = train_state

            if jnp.isnan(train_loss):
                comms.warn(f"loss became NaN at step={self.step}, canceling training")
                self.cancel = True

            if get_lr(opt_state) < min_learning_rate:
                # sometimes we stop decay before max_steps, in that case don't break
                if stop_decay_after == max_epochs:
                    comms.talk(
                        f"learning rate has reached minimum at step={self.step}, canceling"
                    )
                    self.cancel = True

            info = {
                "lr": get_lr(opt_state),
                "time_per_step": self.time_per_step,
                "compute_time_per_step": self.compute_time_per_step,
                "compiles_do_batch": self.compilations,
                **info,
            }

            for logger in self.loggers:
                logger(
                    self.state["step"],
                    train_loss,
                    train_metrics,
                    valid_loss,
                    valid_metrics,
                    other=info,
                )

            metrics = {
                "train": train_metrics,
                "valid": valid_metrics,
            }
            metrics = jax.tree_util.tree_map(lambda x: np.array(x), metrics)

            save_checkpoints(
                metrics,
                params,
                self.state,
                self.model,
                self.baseline,
                self.workdir,
                config=config,
            )

            title = f"state at step: {self.step}"
            msg = []

            msg.append(f"train loss: {train_loss:.5e}")
            msg.append(f"valid loss: {valid_loss:.5e}")

            msg.append(report_on_lr(opt_state))

            msg.append("validation errors:")
            msg += format_metrics(
                metrics["valid"],
                keys=keys,
                properties=properties,
            )

            msg.append("")
            msg.append(f"elapsed: {s2s(self.elapsed, 's')}")
            msg.append(f"timing: {s2s(self.time_per_step)}/step, {s2s(self.eta, 'm')} ETA")

            msg.append("")
            comms.state(msg, title=title)

    manager = Manager(
        state,
        valid_every,
        loggers,
        workdir,
        model,
        baseline,
        max_steps,
    )

    @jax.jit
    def do_batch(carry, batch):
        params, opt_state = carry

        loss_and_aux, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
            params, batch
        )
        loss, aux = loss_and_aux
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss)

        params = optax.apply_updates(params, updates)

        return (params, opt_state), (loss, aux)

    aggregate_loss = np.mean
    aggregate_aux = tree_stack

    # -- train! --

    reporter.step("\U0001f684", spin=False)

    start = monotonic()

    iter_train_with_prefetch = prefetch_to_device(iter_train, 2)

    ran_steps = 0
    train_aux = []
    train_loss = []
    report = None
    last_cache_size = 1
    while True:
        try:
            batch = next(iter_train_with_prefetch)
        except StopIteration:
            comms.talk("exhausted training iterator")
            # break
            manager.cancel = True

        if not manager.done:
            ran_steps += batcher.count_samples(batch)
            (params, opt_state), (loss, aux) = do_batch((params, opt_state), batch)
            current_step = manager.step + ran_steps
            del batch
            reporter.tick(f"{current_step}")

            if do_batch._cache_size() > last_cache_size:
                diff = do_batch._cache_size() - last_cache_size
                last_cache_size = do_batch._cache_size()
                manager.compilations += diff
                comms.talk(f"recompiled at step={current_step} ({manager.compilations})")

            if do_batch._cache_size() > compilation_cache_max_size:
                jax.clear_caches()
                comms.talk(f"cleared caches at step={current_step}")
                last_cache_size = 0

            train_aux.append(aux)
            train_loss.append(loss)

        if report is not None:
            manager.report(*report)
            report = None

        if manager.done:
            break

        if manager.should_validate(manager.step + ran_steps):
            iter_valid_with_prefetch = prefetch_to_device(get_valid_iterator(), 2)

            valid_aux = []
            valid_loss = []
            for i, batch in enumerate(iter_valid_with_prefetch):
                reporter.tick(f"{current_step} (valid {(i + 1) * median_batch_size})")

                loss, aux = loss_fn(params, batch)

                valid_aux.append(aux)
                valid_loss.append(loss)

            train_aux = aggregate_aux(train_aux)
            train_metrics = train_metrics_fn(train_aux)

            valid_aux = tree_stack(valid_aux)
            valid_metrics = valid_metrics_fn(valid_aux)

            train_loss = aggregate_loss(train_loss)
            valid_loss = np.mean(valid_loss)

            report = (
                manager.step + ran_steps,
                params,
                opt_state,
                iter_train.get_state(),
                train_loss,
                train_metrics,
                valid_loss,
                valid_metrics,
                {
                    "compiles_do_batch": do_batch._cache_size(),
                    "compiles_loss_fn": loss_fn._cache_size(),
                },
            )

            train_aux = []
            train_loss = []
            ran_steps = 0

    # -- wrap up --
    from marathon.emit import get_all, plot

    reporter.step("wrapup")

    pred_fn = jax.jit(pred_fn)

    test = {}
    for name, (source, save) in test_datasets.items():
        source = DataSource(source, species_to_weight=species_to_weight)
        batcher = get_batcher(valid=True)

        def it():
            for i in range(len(source)):
                atoms = source[i]
                if all([f.filter(atoms) for f in filters]):
                    sample = to_sample.map(atoms)
                    if filter_empty.filter(sample):
                        yield Record(
                            data=sample,
                            metadata=RecordMetadata(index=i, record_key=i),
                        )

        batches = [b.data for b in batcher(it())]
        test[name] = (batches, save)

    def predict_and_collate(params, batches):
        predictions = {k: [] for k in keys}
        labels = {k: [] for k in keys}

        for batch in batches:
            preds = pred_fn(params, batch)

            for key in keys:
                mask = batch.labels[key + "_mask"]
                if mask.any():
                    predictions[key].append(preds[key][mask])
                    labels[key].append(batch.labels[key][mask])

        final_predictions = {}
        final_labels = {}

        for key in keys:
            final_predictions[key] = np.concatenate(predictions[key])
            final_labels[key] = np.concatenate(labels[key])

        return final_labels, final_predictions

    for f, items in get_all(workdir, state):
        if f.suffix == ".backup":
            continue

        comms.talk(f"working on {f}")

        params, _, _, _, metrics, _ = items

        for name, (batches, save) in test.items():
            labels, predictions = predict_and_collate(params, batches)

            out = f / f"plot/{name}"
            out.mkdir(parents=True, exist_ok=True)

            plot(out, predictions, labels)

            if save:
                for key in keys:
                    np.savez_compressed(out / f"{key}.npz", predictions[key])

    reporter.done()
    if use_wandb:
        run.finish()

    comms.talk("cleaning up")
    import shutil

    if use_wandb:
        wandb_dir = Path("wandb")
        if wandb_dir.is_dir():
            shutil.rmtree(wandb_dir)

    for f, items in get_all(workdir, state):
        if f.suffix == ".backup":
            shutil.rmtree(f)

    comms.state("done!")


if __name__ == "__main__":
    main()
