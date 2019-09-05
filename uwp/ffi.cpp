#include "pch.h"
#include "ffi.h"

#include <Windows.h>
#include <string>

extern "C"
{
    struct EvalResult
    {
        double ApproximateResult;
        bool Success;
    };

    EvalResult calculator_eval(const char* expr);

    // fix Rust for UWP
    BOOL GetUserProfileDirectoryW(HANDLE, LPWSTR, LPDWORD)
    {
        return FALSE;
    }
}

bool Calculator::Eval(const winrt::hstring& expr, double& approximateResultOut)
{
    std::string expr_c = winrt::to_string(expr);

    EvalResult result = calculator_eval(expr_c.c_str());
    if (!result.Success)
    {
        return false;
    }

    approximateResultOut = result.ApproximateResult;
    return true;
}
