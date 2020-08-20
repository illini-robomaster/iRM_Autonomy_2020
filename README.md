# RM_2020_DEV
iRM 2020 Software Development Repo

## Dependency
* Glib2 (for LCM)
* CMake >= 3.12

## Build Steps
```
cd <path to RM_2020DEV>
mkdir build && cd build
cmake ..
make -j
```

## Run Tests
`make test` or `ctest` in `build` folder
