import csv
from datetime import datetime
from pathlib import Path


class Log:
    def __init__(self):
        self._dir = None
        self._log_file = None

    def setup(self, dir, log_filename='log.txt', if_exists='append'):
        assert self._dir is None, 'Can only setup once'
        self._dir = Path(dir)
        self._dir.mkdir(exist_ok=True)
        print(f'Set log dir to {dir}, log filename to {log_filename}')
        log_path = self._dir / log_filename
        if log_path.exists():
            if if_exists == 'append':
                print(f'Log file named {log_filename} already exists; will use append mode')
                self._log_file = log_path.open('a', buffering=1)
            elif if_exists == 'overwrite':
                print(f'Log file named {log_filename} already exists; will overwrite it')
                self._log_file = log_path.open('w', buffering=1)
            elif if_exists == 'exit':
                print(f'Log file named {log_filename} already exists; exiting')
                exit()
            else:
                raise NotImplementedError(f'Unknown if_exists option: {if_exists}')
        else:
            print(f'Creating new log file named {log_filename}')
            self._log_file = log_path.open('w', buffering=1)

    @property
    def dir(self):
        return self._dir

    def message(self, message, timestamp=True, flush=False):
        if timestamp:
            now_str = datetime.now().strftime('%H:%M:%S')
            message = f'[{now_str}] ' + message
        else:
            message = ' ' * 11 + message
        print(message)
        self._log_file.write(f'{message}\n')
        if flush:
            self._log_file.flush()

    def __call__(self, *args, **kwargs):
        return self.message(*args, **kwargs)


default_log = Log()


class TabularLog:
    def __init__(self, dir, filename):
        self._dir = Path(dir)
        assert self._dir.is_dir()
        self._filename = filename
        self._column_names = None
        self._file = open(self.path, mode=('a' if self.path.exists() else 'w'), newline='')
        self._writer = csv.writer(self._file)

    @property
    def path(self):
        return self._dir/self._filename

    def row(self, row, flush=True):
        if self._column_names is None:
            self._column_names = list(row.keys())
            self._writer.writerow(self._column_names)
        self._writer.writerow([row[col] for col in self._column_names])
        if flush:
            self._file.flush()