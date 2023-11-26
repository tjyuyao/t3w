from __future__ import annotations

# fmt:off
import aim

from aim.sdk.repo import Repo
from aim.sdk.run import get_installed_packages, get_environment_variables, get_git_info, sys
from aim.sdk.run import Optional, BasicRunAutoClean, TreeView, logger, backup_run, RunStatusReporter, LocalFileManager, pathlib, RemoteFileManager, ScheduledStatusReporter, RemoteRunHeartbeatReporter, RunTracker
from aim.sdk.base_run import get_repo, analytics, MissingRunError, Dict
from aim.ext.resource import ResourceTracker, DEFAULT_SYSTEM_TRACKING_INT

from t3w.utils.misc import generate_run_hash as t3w_generate_run_hash


# [feat] Allow users to specify the run_hash: https://github.com/aimhubio/aim/pull/2738/commits/fd48e0a3dc7e505fff86791dd30d70b3d899b008
class Run(aim.Run):

    def __init__(self, run_hash: str | None = None, *, repo: str | Repo | pathlib.Path | None = None, read_only: bool = False, experiment: str | None = None, force_resume: bool = False, system_tracking_interval: int | float | None = 600, log_system_params: bool | None = False, capture_terminal_logs: bool | None = True):

        self._resources: Optional[BasicRunAutoClean] = None

        #########################
        # aim.BaseRun.__init__  #
        #########################

        self._hash = None
        self._lock = None

        self.read_only = read_only
        self.repo = get_repo(repo)
        if self.read_only:
            assert run_hash is not None
            self.hash = run_hash
        else:
            if run_hash is None or not self.repo.run_exists(run_hash):
                self.hash = run_hash or t3w_generate_run_hash()
            # if run_hash is None:
            #     self.hash = generate_run_hash()
                analytics.track_event(event_name='[Run] Create new run')
            else:
            # elif self.repo.run_exists(run_hash):
                self.hash = run_hash
                analytics.track_event(event_name='[Run] Resume run')
            # else:
            #     raise MissingRunError(f'Cannot find Run {run_hash} in aim Repo {self.repo.path}.')
            self._lock = self.repo.request_run_lock(self.hash)
            self._lock.lock(force=force_resume)

        self.meta_tree: TreeView = self.repo.request_tree(
            'meta', self.hash, read_only=read_only, from_union=True
        ).subtree('meta')
        self.meta_run_tree: TreeView = self.meta_tree.subtree('chunks').subtree(self.hash)

        self._series_run_trees: Dict[int, TreeView] = None

        #########################
        # aim.BasicRun.__init__ #
        #########################

        self.meta_attrs_tree: TreeView = self.meta_tree.subtree('attrs')
        self.meta_run_attrs_tree: TreeView = self.meta_run_tree.subtree('attrs')

        if not read_only:
            logger.debug(f'Opening Run {self.hash} in write mode')

            if self.check_metrics_version():
                if self.repo.is_remote_repo:
                    logger.warning(f'Cannot track Run with remote repo {self.repo.path}. Please upgrade repo first '
                                   f'with the following command:')
                    logger.warning(f'aim storage --repo {self.repo.path} upgrade 3.11+ \'*\'')
                    raise RuntimeError
                else:
                    logger.warning(f'Detected sub-optimal format metrics for Run {self.hash}. Upgrading...')
                    backup_path = backup_run(self)
                    try:
                        self.update_metrics()
                        logger.warning(f'Successfully converted Run {self.hash}')
                        logger.warning(f'Run backup can be found at {backup_path}. '
                                       f'In case of any issues the following command can be used to restore data: '
                                       f'`aim storage --repo {self.repo.root_path} restore {self.hash}`')
                    except Exception as e:
                        logger.error(f'Failed to convert metrics. {e}')
                        logger.warning(f'Run backup can be found at {backup_path}. '
                                       f'To restore data please run the following command: '
                                       f'`aim storage --repo {self.repo.root_path} restore {self.hash}`')
                        raise

        self._props = None
        self._checkins = None
        self._heartbeat = None

        if not read_only:
            if not self.repo.is_remote_repo:
                self._checkins = RunStatusReporter(self.hash, LocalFileManager(self.repo.path))
                progress_flag_path = pathlib.Path(self.repo.path) / 'meta' / 'progress' / self.hash
                self._heartbeat = ScheduledStatusReporter(self._checkins, touch_path=progress_flag_path)
            else:
                self._checkins = RunStatusReporter(self.hash, RemoteFileManager(self.repo._client, self.hash))
                self._heartbeat = RemoteRunHeartbeatReporter(self.repo._client, self.hash)

            try:
                self.meta_run_attrs_tree.first_key()
            except (KeyError, StopIteration):
                # no run params are set. use empty dict
                self[...] = {}
            self.meta_run_tree['end_time'] = None
            self.props
        if experiment:
            self.experiment = experiment

        self._tracker = RunTracker(self)
        self._resources = BasicRunAutoClean(self)

        #####################
        # aim.Run.__init__  #
        #####################

        self._system_resource_tracker: ResourceTracker = None
        if not read_only:
            if log_system_params:
                self['__system_params'] = {
                    'packages': get_installed_packages(),
                    'env_variables': get_environment_variables(),
                    'git_info': get_git_info(),
                    'executable': sys.executable,
                    'arguments': sys.argv
                }

            if ResourceTracker.check_interval(system_tracking_interval) or capture_terminal_logs:
                current_logs = self.get_terminal_logs()
                log_offset = current_logs.last_step() + 1 if current_logs else 0
                self._system_resource_tracker = ResourceTracker(self._tracker,
                                                                system_tracking_interval,
                                                                capture_terminal_logs,
                                                                log_offset)
                self._system_resource_tracker.start()
                self._resources.add_extra_resource(self._system_resource_tracker)

# fmt:on
