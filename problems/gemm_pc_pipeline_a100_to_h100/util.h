#pragma once
#include <stddef.h>
inline size_t div_ceil(size_t a, size_t b) { return (a + b - 1) / b; }
