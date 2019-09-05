#pragma once

#include "MainPage.g.h"

#include <optional>
#include <mutex>

namespace winrt::Calculator::implementation
{
    struct MainPage : MainPageT<MainPage>
    {
        MainPage();

        void InputChangedHandler(Windows::Foundation::IInspectable const& sender, Windows::UI::Xaml::Controls::TextChangedEventArgs const& args);

    private:
        Windows::Foundation::IAsyncAction _evalThread;

        std::mutex _evalQueueMutex;
        std::optional<winrt::hstring> _evalQueue;
    };
}

namespace winrt::Calculator::factory_implementation
{
    struct MainPage : MainPageT<MainPage, implementation::MainPage>
    {
    };
}
