#include "pch.h"
#include "MainPage.h"
#include "MainPage.g.cpp"

#include <sstream>
#include <stdexcept>

#include "ffi.h"

using namespace winrt;
using namespace Windows::UI::Core;
using namespace Windows::UI::Xaml;
using namespace Windows::System::Threading;
using namespace Windows::Foundation;
using namespace Windows::ApplicationModel::Core;

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

        {
            std::lock_guard g(_evalQueueMutex);

            _evalQueue = { std::move(text) };
        }

        if (!_evalThread || _evalThread.Status() != AsyncStatus::Started)
        {
            _evalThread = ThreadPool::RunAsync([&](IAsyncAction const&)
                {
                    while (true)
                    {
                        hstring expr;

                        {
                            std::lock_guard g(_evalQueueMutex);

                            if (!_evalQueue)
                            {
                                break;
                            }

                            // get and remove next expression from the queue
                            expr = std::move(*_evalQueue);
                            _evalQueue = std::nullopt;
                        }

                        double approximateResult;
                        bool success = ::Calculator::Eval(expr, approximateResult);

                        auto updateAction = CoreApplication::MainView().CoreWindow().Dispatcher()
                            .RunAsync(CoreDispatcherPriority::High, [success, approximateResult, this]()
                                {
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
                                });

                        updateAction.get();
                    }
                });
        }
        else
        {
        }
    }
}
