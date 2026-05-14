# Introduction to Interop

## Training

```bas
NoteBook Jupyter (Python)                   Lib (C, C++, Rust, Zig)
            |_                                          |
            | \______  create_linear_model(nb_inputs)   |
            |        \______                            |
            |               \______                     |
            |                      \______              |
            |                             \______       |
            |                                    \______|
            |                                    ______/|
            |                *Model       ______/       |
            |                      ______/              |
            |               ______/                     |
            |        ______/                            |
            | ______/                                   |
            |/                                          |
            |_                                          |
            | \______  train_vosenblattrole(dataset, epochs, alpha, pkmodel)
            |        \______                            |
            |               \______                     |
            |                      \______              |
            |                             \______       |
            |                                    \______|
            |                                    ______/|
            |                matrices     ______/       |
            |                      ______/              |
            |               ______/                     |
            |        ______/                            |
            | ______/                                   |
            |/                                          |
            |_                                          |
            | \______            train                  |
            |        \______                            |
            |               \______                     |
            |                      \______              |
            |                             \______       |
            |                                    \______|
            |                                    ______/|
            |                .........    ______/       |
            |                      ______/              |
            |               ______/                     |
            |        ______/                            |
            | ______/                                   |
            |/                                          |
            |_                                          |
            | \______            train                  |
            |        \______                            |
            |               \______                     |
            |                      \______              |
            |                             \______       |
            |                                    \______|
            |                                    ______/|
            |                .........    ______/       |
            |                      ______/              |
            |               ______/                     |
            |        ______/                            |
            | ______/                                   |
            |/                                          |
            |_                                          |
            | \______ save_linear_model(*model, "path") |
            |        \______                            |
            |               \______                     |
            |                      \______              |
            |                             \______       |
            |                                    \______|
            |                                    ______/|
            |                .........    ______/       |
            |                      ______/              |
            |               ______/                     |
            |        ______/                            |
            | ______/                                   |
            |/                                          |
```
        

## Inference

```bash
Server Web                            Lib (C, C++, Rust, Zig)
    |_                                          |
    | \______  load_linear_model("path")        |
    |        \______                            |
    |               \______                     |
    |                      \______              |
    |                             \______       |
    |                                    \______|
    |                                    ______/|
    |                *Model       ______/       |
    |                      ______/              |
    |               ______/                     |
    |        ______/                            |
    | ______/                                   |
    |/                                          |
    |_                                          |
    | \______  predict_linear_model(model, inputs)
    |        \______                            |
    |               \______                     |
    |                      \______              |
    |                             \______       |
    |                                    \______|
    |                                    ______/|
    |                outputs      ______/       |
    |                      ______/              |
    |               ______/                     |
    |        ______/                            |
    | ______/                                   |
    |/                                          |
    |_                                          |
    | \______            Predict...             |
    |        \______                            |
    |               \______                     |
    |                      \______              |
    |                             \______       |
    |                                    \______|
    |                                    ______/|
    |                .........    ______/       |
    |                      ______/              |
    |               ______/                     |
    |        ______/                            |
    | ______/                                   |
    |/                                          |
    |_                                          |
    | \______            Predict...             |
    |        \______                            |
    |               \______                     |
    |                      \______              |
    |                             \______       |
    |                                    \______|
    |                                    ______/|
    |                .........    ______/       |
    |                      ______/              |
    |               ______/                     |
    |        ______/                            |
    | ______/                                   |
    |/                                          |
    |_                                          |
    | \______ destroy_linear_model(*model)      |
    |        \______                            |
    |               \______                     |
    |                      \______              |
    |                             \______       |
    |                                    \______|
```


## How to make code cross compilable

This will not cross compile and not work with Python :
```cpp
int my_add(int a, int b){
    return a + b;
}
```

This will cross compile with Python on Linux and Mac (not Windows yet):
```cpp
extern "C" {
    int my_add(int a, int b){
        return a + b;
    }
}
```

This will cross compile with Python on Windows (not Linux and Mac)
```cpp
extern "C" {
    __declspec(dllexport) int my_add(int a, int b) {
        return a + b;
    }
}
```

This will cross compile with python on Linux, Mac and Windows
```cpp
#ifdef WIN32
#define DLLEXPORT __delspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {
    DLLEXPORT int my_add(int a, int b) {
        return a + b;
    }
}
```

## Python example of library calling

```py
import ctypes

def main():
    lib_cpp = ctypes.cdll.LoadLibrary("/home/leo/Documents/ESGI/PA/test.dll")
    lib_cpp.my_add.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib_cpp.my_add.restype = ctypes.c_int32

    lib_cpp.my_array_sum.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    lib_cpp.my_array_sum.restype = ctypes.c_int32

    print(lib_cpp.my_add(*args:32, 67))
```
