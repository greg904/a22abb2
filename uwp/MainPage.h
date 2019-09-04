#pragma once

#include "MainPage.g.h"

namespace winrt::Calculator::implementation
{
    struct MainPage : MainPageT<MainPage>
    {
        MainPage();

        void InputChangedHandler(Windows::Foundation::IInspectable const& sender, Windows::UI::Xaml::Controls::TextChangedEventArgs const& args);
    };
}

namespace winrt::Calculator::factory_implementation
{
    struct MainPage : MainPageT<MainPage, implementation::MainPage>
    {
    };
}
