using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace A22abb2
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public ObservableCollection<CalculationViewModel> History { get; set; } = new ObservableCollection<CalculationViewModel>();

        public CalculationViewModel CurrentCalculation { get; } = new CalculationViewModel();

        public MainPage()
        {
            this.InitializeComponent();

            CurrentCalculation.PropertyChanged += (sender, e) =>
            {
                if (e.PropertyName == "Expression")
                {
                    string newResult = null;

                    if (!CurrentCalculation.Expression.All(char.IsWhiteSpace))
                    {
                        // TODO: do not call this on the UI thread
                        var result = A22abb2.Eval(CurrentCalculation.Expression);
                        if (!double.IsNaN(result))
                        {
                            newResult = result.ToString();
                        }
                    }

                    CurrentCalculation.Result = newResult;
                }
            };
        }

        private void AddHistory_Click(object sender, RoutedEventArgs e)
        {
            StoreCurrentExpressionInHistory();
        }

        private void History_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            foreach (var item in e.AddedItems)
            {
                var modelItem = (CalculationViewModel)item;
                if (modelItem != null)
                {
                    // put it back in the current calculation
                    StoreCurrentExpressionInHistory();
                    CurrentCalculation.Expression = modelItem.Expression;
                    History.Remove(modelItem);
                }
            }
        }

        private void StoreCurrentExpressionInHistory()
        {
            if (CurrentCalculation.Result != null)
            {
                History.Add((CalculationViewModel)CurrentCalculation.Clone());
                CurrentCalculation.Reset();
            }
        }

        private void DeleteButton_Click(object sender, RoutedEventArgs e)
        {
            var button = (Button)sender;
            var item = (CalculationViewModel)button.DataContext;
            History.Remove(item);
        }
    }
}
