#include "pch.h"
#include "MainPage.h"
#include "MainPage.g.cpp"

#include <sstream>

#include "ffi.h"

using namespace winrt;
using namespace Windows::UI::Xaml;

namespace winrt::Calculator::implementation
{
    MainPage::MainPage()
    {
        InitializeComponent();

        ApproximateResult().Visibility(Visibility::Collapsed);
    }

    void MainPage::InputChangedHandler(IInspectable const&, Controls::TextChangedEventArgs const&)
    {
        hstring text = InputBox().Text();

        double approximateResult;
        bool success = ::Calculator::Eval(text, approximateResult);

        if (success)
        {
            std::wostringstream ss;
            ss << L"≅ ";
            ss << approximateResult;

            ApproximateResult().Text(ss.str());
            ApproximateResult().Visibility(Visibility::Visible);
        }
        else
        {
            ApproximateResult().Visibility(Visibility::Collapsed);
        }
    }
}
