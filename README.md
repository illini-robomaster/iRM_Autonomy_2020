# RM_2020_DEV
iRM 2020 Software Development Repo

## Dependency
* Boost
* CMake >= 3.12
* GTest
* [lcm](http://lcm-proj.github.io/)

## Build Steps
```
cd <path to RM_2020DEV>
mkdir build && cd build
cmake ..
make -j
```

## Run Tests
`make test` or `ctest` in `build` folder
