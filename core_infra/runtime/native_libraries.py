"""Pin primary native libraries before loading private model closures."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


def absolute(path):
    if (type(path) is not str or not Path(path).is_absolute() or '..' in Path(path).parts or
            any(character.isspace() or character in ':\0' for character in path)):
        raise ValueError('native loader paths must be absolute unambiguous paths')
    return path


@dataclass(frozen=True)
class LibraryPin:
    path: str
    sha256: str

    def verify(self):
        absolute(self.path)
        if not re.fullmatch('[0-9a-f]{64}', self.sha256):
            raise ValueError('native preload requires an immutable SHA256')
        with Path(self.path).open('rb') as source:
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
        if actual != self.sha256:
            raise RuntimeError('native primary library changed: ' + self.path)


@dataclass(frozen=True)
class NativeLibraryEnvironment:
    abi_directories: tuple[str, ...] = ()
    primary_directories: tuple[str, ...] = ()
    private_directories: tuple[str, ...] = ()
    primary_preloads: tuple[LibraryPin, ...] = ()

    def apply(self, base):
        paths = []
        for directories in (self.abi_directories, self.primary_directories, self.private_directories):
            if type(directories) is not tuple:
                raise ValueError('native library declarations must be immutable')
            paths.extend(absolute(path) for path in directories)
        if type(self.primary_preloads) is not tuple or any(not isinstance(pin, LibraryPin) for pin in self.primary_preloads):
            raise ValueError('native preloads require immutable library pins')
        for pin in self.primary_preloads:
            pin.verify()
        existing = [absolute(path) for path in re.split('[:\\s]+', base.get('LD_PRELOAD', '')) if path]
        environment = dict(base)
        environment['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(paths))
        preloads = list(dict.fromkeys([pin.path for pin in self.primary_preloads] + existing))
        if preloads:
            environment['LD_PRELOAD'] = ':'.join(preloads)
        return environment
