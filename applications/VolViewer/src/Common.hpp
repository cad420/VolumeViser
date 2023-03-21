#pragma once

#include <wrl/client.h>
#include <d3d11_1.h> //use 11.0 is ok
#include <d3dcompiler.h>

template <class T>
using ComPtr = Microsoft::WRL::ComPtr<T>;