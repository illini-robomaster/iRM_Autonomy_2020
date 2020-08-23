# RM_2020_DEV 

![build status](https://github.com/illini-robomaster/iRM_Autonomy_2020/workflows/IRM%20Autonomy%20CI/badge.svg)

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
