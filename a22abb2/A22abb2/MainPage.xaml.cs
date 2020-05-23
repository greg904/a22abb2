using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows.Input;
using Windows.System;
using Windows.UI.Xaml.Controls;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace A22abb2
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public readonly MainViewModel ViewModel = new MainViewModel();

        public MainPage()
        {
            this.InitializeComponent();
        }

        private void ExpressionTextBox_KeyDown(object sender, Windows.UI.Xaml.Input.KeyRoutedEventArgs e)
        {
            if (e.Key == VirtualKey.Enter)
            {
                this.ViewModel.CommitIfValid();
                e.Handled = true;
            }
        }
    }

    public sealed class MainViewModel : INotifyPropertyChanged
    {
        public readonly ObservableCollection<CalculationViewModel> History = new ObservableCollection<CalculationViewModel>();

        private CalculationViewModel currentCalculation = new CalculationViewModel();
        public CalculationViewModel CurrentCalculation
        {
            get => this.currentCalculation;
            set
            {
                if (this.currentCalculation != value)
                {
                    this.currentCalculation = value;
                    this.OnPropertyChanged(nameof(CurrentCalculation));
                }
            }
        }

        public readonly CommitCommand CommitCommand;

        public event PropertyChangedEventHandler PropertyChanged;

        public MainViewModel()
        {
            this.CommitCommand = new CommitCommand(this);
            this.History.Add(this.CurrentCalculation);
        }

        public void CommitIfValid()
        {
            if (double.IsNaN(this.CurrentCalculation.Result))
            {
                return;
            }
            this.CurrentCalculation = new CalculationViewModel();
            this.History.Add(this.CurrentCalculation);
        }

        private void OnPropertyChanged(string name)
        {
            this.PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }

    public sealed class CommitCommand : ICommand
    {
        public event EventHandler CanExecuteChanged;
        private readonly MainViewModel viewModel;

        public CommitCommand(MainViewModel viewModel)
        {
            this.viewModel = viewModel;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            this.viewModel.CommitIfValid();
        }
    }
}
