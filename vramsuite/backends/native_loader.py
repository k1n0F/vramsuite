from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
from dataclasses import  dataclass
from pathlib import Path
from types import TracebackType



class NativeLibraryLoadError(Exception):
    pass

@dataclass
class LoadedNativeLibrary:
    logical_name: str
    candidates: list[str]
    venv_patterns: list[str]

    library: ctypes.CDLL | None = None
    library_path: str | None = None
    error: str | None = None

    _dll_directory_handle: object | None = None

    def __enter__(self) -> "LoadedNativeLibrary":
        self.library, self.library_path = self._load()
        return self
    
    def __exit__(
        self,
        exc_types: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.library = None
        self.library_path = None
        self.error = None

        self._dll_directory_handle = None

    def require_library(self) -> ctypes.CDLL:
        if self.library is None:
            raise NativeLibraryLoadError(
                f"{self.logical_name} native library is not loaded"
            )
        
        return self.library
    
    def _venv_candidates(self) -> list[str]:
        venv = Path(".venv")

        if not venv.exists():
            return []
        
        results: list[str] = []


        for pattern in self.venv_patterns:
            results.extend(str(path) for path in venv.rglob(pattern))

        return results
        
    def _build_candidate_list(self) -> list[str]:
        results: list[str] = []

        found = ctypes.util.find_library(self.logical_name)
        if found:
            results.append(found)

        results.extend(self.candidates)
        results.extend(self._venv_candidates())

        unique: list[str] = []
        seen: set[str] = set()

        for item in results:
            if item in seen:
                continue

            unique.append(item)
            seen.add(item)

        return unique
    
    def _load(self) -> tuple[ctypes.CDLL, str]:
        errors: list[str] = []

        for candidate in self._build_candidate_list():
            try:
                path = Path(candidate)

                if path.exists():
                    resolved_path = path.resolve()

                    if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
                        self._dll_directory_handle = os.add_dll_directory(
                            str(resolved_path.parent)
                        )

                    return ctypes.CDLL(str(resolved_path)), str(resolved_path)
                
                return ctypes.CDLL(candidate), candidate
            
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")

        message = (
            f"{self.logical_name} native library could not found. Tried: "
            + "; ".join(errors)
        )

        self.error = message
        raise NativeLibraryLoadError(message)


def load_native_library(
        logical_name: str,
        candidates: list[str],
        venv_patterns: list[str] | None = None,
) -> LoadedNativeLibrary:
    return LoadedNativeLibrary(
        logical_name=logical_name,
        candidates=candidates,
        venv_patterns=venv_patterns or [],
    )
        