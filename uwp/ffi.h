#pragma once

#include <winrt/base.h>

namespace Calculator
{
    bool Eval(const winrt::hstring& expr, double& approximateResultOut);
}
