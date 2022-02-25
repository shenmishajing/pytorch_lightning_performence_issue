# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional, Union

Task, Style = None, None
from rich.console import Console, RenderableType
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskID, TextColumn
from rich.progress_bar import ProgressBar
from rich.style import Style
from rich.text import Text

from mmcv.runner.hooks import LoggerHook, HOOKS


class CustomBarColumn(BarColumn):
    """Overrides ``BarColumn`` to provide support for dataloaders that do not define a size (infinite size)
    such as ``IterableDataset``."""

    def render(self, task: "Task") -> ProgressBar:
        """Gets a progress bar widget for a task."""
        return ProgressBar(
            total = max(0, task.total),
            completed = max(0, task.completed),
            width = None if self.bar_width is None else max(1, self.bar_width),
            pulse = not task.started or not math.isfinite(task.remaining),
            animation_time = task.get_time(),
            style = self.style,
            complete_style = self.complete_style,
            finished_style = self.finished_style,
            pulse_style = self.pulse_style,
        )


@dataclass
class CustomInfiniteTask(Task):
    """Overrides ``Task`` to define an infinite task.

    This is useful for datasets that do not define a size (infinite size) such as ``IterableDataset``.
    """

    @property
    def time_remaining(self) -> Optional[float]:
        return None


class CustomProgress(Progress):
    """Overrides ``Progress`` to support adding tasks that have an infinite total size."""

    def add_task(
            self,
            description: str,
            start: bool = True,
            total: float = 100.0,
            completed: int = 0,
            visible: bool = True,
            **fields: Any,
    ) -> TaskID:
        if not math.isfinite(total):
            task = CustomInfiniteTask(
                self._task_index,
                description,
                total,
                completed,
                visible = visible,
                fields = fields,
                _get_time = self.get_time,
                _lock = self._lock,
            )
            return self.add_custom_task(task)
        return super().add_task(description, start, total, completed, visible, **fields)

    def add_custom_task(self, task: CustomInfiniteTask, start: bool = True):
        with self._lock:
            self._tasks[self._task_index] = task
            if start:
                self.start_task(self._task_index)
            new_task_index = self._task_index
            self._task_index = TaskID(int(self._task_index) + 1)
        self.refresh()
        return new_task_index


class CustomTimeColumn(ProgressColumn):
    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def __init__(self, style: Union[str, Style]) -> None:
        self.style = style
        super().__init__()

    def render(self, task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        remaining = task.time_remaining
        elapsed_delta = "-:--:--" if elapsed is None else str(timedelta(seconds = int(elapsed)))
        remaining_delta = "-:--:--" if remaining is None else str(timedelta(seconds = int(remaining)))
        return Text(f"{elapsed_delta} • {remaining_delta}", style = self.style)


class BatchesProcessedColumn(ProgressColumn):
    def __init__(self, style: Union[str, Style]):
        self.style = style
        super().__init__()

    def render(self, task) -> RenderableType:
        total = task.total if task.total != float("inf") else "--"
        return Text(f"{int(task.completed)}/{total}", style = self.style)


class ProcessingSpeedColumn(ProgressColumn):
    def __init__(self, style: Union[str, Style]):
        self.style = style
        super().__init__()

    def render(self, task) -> RenderableType:
        task_speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
        return Text(f"{task_speed}it/s", style = self.style)


class MetricsTextColumn(ProgressColumn):
    """A column containing text."""

    def __init__(self, runner, style):
        self._runner = runner
        self._tasks = {}
        self._current_task_id = 0
        self._metrics = {}
        self._style = style
        super().__init__()

    def update(self, metrics):
        # Called when metrics are ready to be rendered.
        # This is to prevent render from causing deadlock issues by requesting metrics
        # in separate threads.
        self._metrics = metrics

    def render(self, task) -> Text:
        if self._runner.mode != 'train':
            return Text("")
        if task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
            self._current_task_id = task.id
        if task.id != self._current_task_id:
            return self._tasks[task.id]
        _text = ""

        for k, v in self._metrics.items():
            _text += f"{k}: {round(v, 3) if isinstance(v, float) else v} "
        return Text(_text, justify = "left", style = self._style)


@dataclass
class RichProgressBarTheme:
    """Styles to associate to different base components.

    Args:
        description: Style for the progress bar description. For eg., Epoch x, Testing, etc.
        progress_bar: Style for the bar in progress.
        progress_bar_finished: Style for the finished progress bar.
        progress_bar_pulse: Style for the progress bar when `IterableDataset` is being processed.
        batch_progress: Style for the progress tracker (i.e 10/50 batches completed).
        time: Style for the processed time and estimate time remaining.
        processing_speed: Style for the speed of the batches being processed.
        metrics: Style for the metrics

    https://rich.readthedocs.io/en/stable/style.html
    """

    description: Union[str, Style] = "white"
    progress_bar: Union[str, Style] = "#6206E0"
    progress_bar_finished: Union[str, Style] = "#6206E0"
    progress_bar_pulse: Union[str, Style] = "#6206E0"
    batch_progress: Union[str, Style] = "white"
    time: Union[str, Style] = "grey54"
    processing_speed: Union[str, Style] = "grey70"
    metrics: Union[str, Style] = "white"


@HOOKS.register_module()
class RichProgressBarHook(LoggerHook):
    """Create a progress bar with `rich text formatting <https://github.com/willmcgugan/rich>`_.

    Install it with pip:

    .. code-block:: bash

        pip install rich

    Args:
        refresh_rate_per_second: the number of updates per second. If refresh_rate is 0, progress bar is disabled.
        leave: Leaves the finished progress bar in the terminal at the end of the epoch. Default: False
        theme: Contains styles used to stylize the progress bar.

    Raises:
        ModuleNotFoundError:
            If required `rich` package is not installed on the device.
    """

    def __init__(
            self,
            refresh_rate_per_second: int = 10,
            leave: bool = False,
            metric_key_map: dict[str, str] = None,
            theme: RichProgressBarTheme = RichProgressBarTheme(),
            *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.time_sec_tot = 0
        self.start_iter = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self._refresh_rate_per_second: int = refresh_rate_per_second
        self._leave: bool = leave
        self._enabled: bool = True
        self.progress: Optional[Progress] = None
        self.val_sanity_progress_bar_id: Optional[int] = None
        self._reset_progress_bar_ids()
        self._metric_component = None
        self._progress_stopped: bool = False
        self.theme = theme

        if metric_key_map is None:
            metric_key_map = {
                'train/loss': 'loss',
                'loss': None,
            }
        self.metric_key_map = metric_key_map

    @property
    def refresh_rate_per_second(self) -> float:
        """Refresh rate for Rich Progress.

        Returns: Refresh rate for Progress Bar.
            Return 1 if not enabled, as a positive integer is required (ignored by Rich Progress).
        """
        return self._refresh_rate_per_second if self._refresh_rate_per_second > 0 else 1

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._refresh_rate_per_second > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    @property
    def validation_description(self) -> str:
        return "Validation"

    def _init_progress(self, runner):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            self._console: Console = Console()
            self._console.clear_live()
            self._metric_component = MetricsTextColumn(runner, self.theme.metrics)
            self.progress = CustomProgress(
                *self.configure_columns(),
                self._metric_component,
                refresh_per_second = self.refresh_rate_per_second,
                disable = self.is_disabled,
                console = self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False

    def _stop_progress(self) -> None:
        if self.progress is not None:
            self.progress.stop()
            # # signals for progress to be re-initialized for next stages
            self._progress_stopped = True

    def before_run(self, runner):
        super().before_run(runner)
        self.start_iter = runner.iter
        self._init_progress(runner)
        runner.progress_bar = self

    def before_epoch(self, runner):
        super().before_epoch(runner)
        total_batches = len(runner.data_loader)
        if runner.mode == 'train':
            description = f"Epoch {runner.epoch + 1}/{runner._max_epochs}"
        else:
            description = self.validation_description
        if self.main_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(runner)
        if self.main_progress_bar_id is None:
            self.main_progress_bar_id = self.add_task(total_batches, description)
        elif self.progress is not None:
            self.progress.reset(
                self.main_progress_bar_id, total = total_batches, description = description, visible = True
            )

    def after_iter(self, runner):
        self.update(self.main_progress_bar_id)

    def after_train_iter(self, runner):
        super().after_train_iter(runner)
        super(LoggerHook, self).after_train_iter(runner)

    def after_epoch(self, runner):
        if self.get_epoch(runner) >= runner._max_epochs:
            self._stop_progress()

    def after_train_epoch(self, runner):
        super().after_train_epoch(runner)
        super(LoggerHook, self).after_train_epoch(runner)

    def after_val_epoch(self, runner):
        super().after_val_epoch(runner)
        super(LoggerHook, self).after_val_epoch(runner)

    def log(self, runner):
        self._update_metrics(runner)

    def __getstate__(self):
        # can't pickle the rich progress objects
        state = self.__dict__.copy()
        state["progress"] = None
        state["_console"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        state["_console"] = Console()

    def add_task(self, total_batches: int, description: str, visible: bool = True) -> Optional[int]:
        if self.progress is not None:
            return self.progress.add_task(
                f"[{self.theme.description}]{description}", total = total_batches, visible = visible
            )

    def update(self, progress_bar_id: int, visible: bool = True) -> None:
        if self.progress is not None:
            self.progress.update(progress_bar_id, advance = 1.0, visible = visible)

    def _reset_progress_bar_ids(self):
        self.main_progress_bar_id: Optional[int] = None

    def _update_metrics(self, runner) -> None:
        metrics = self.get_loggable_tags(runner)
        metrics = {k if v is None else v: metrics[k] for k, v in self.metric_key_map.items() if k in metrics}
        if 'time' in runner.log_buffer.output:
            self.time_sec_tot += runner.log_buffer.output['time'] * self.interval
            time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(timedelta(seconds = int(eta_sec)))
            metrics['ETA'] = eta_str
        if self._metric_component:
            self._metric_component.update(metrics)

    def configure_columns(self) -> list:
        return [
            TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                complete_style = self.theme.progress_bar,
                finished_style = self.theme.progress_bar_finished,
                pulse_style = self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style = self.theme.batch_progress),
            CustomTimeColumn(style = self.theme.time),
            ProcessingSpeedColumn(style = self.theme.processing_speed),
        ]
