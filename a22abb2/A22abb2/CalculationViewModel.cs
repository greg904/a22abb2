using System.ComponentModel;
using System.Text.RegularExpressions;
using Windows.ApplicationModel.Resources;
using Windows.UI.Xaml;

namespace A22abb2
{
    public sealed class CalculationViewModel : INotifyPropertyChanged
    {
        private static readonly Regex WhitespaceRegex = new Regex(@"\s+");

        private string expression = "";

        private Ffi.EvalResult result;

        public event PropertyChangedEventHandler PropertyChanged;

        public Visibility HistoryVisibility { get; private set; } = Visibility.Collapsed;
        public Visibility SimplificationVisibility { get; private set; } = Visibility.Collapsed;
        public Visibility ApproximationVisibility { get; private set; } = Visibility.Collapsed;

        public string Expression
        {
            get => this.expression;
            set
            {
                if (this.expression != value)
                {
                    this.expression = value;
                    this.OnPropertyChanged(nameof(Expression));

                    // TODO: do not call this on the UI thread
                    var newResult = Ffi.Eval(this.expression.Trim());
                    if (!this.result.Equals(newResult))
                    {
                        this.result = newResult;
                        this.OnPropertyChanged(nameof(HasFailed));
                        this.OnPropertyChanged(nameof(Approximation));
                        this.OnPropertyChanged(nameof(SimplifiedExpression));
                    }

                    var original = WhitespaceRegex.Replace(this.Expression, ""); ;
                    var simplified = this.result.SimplifiedExpression == null ?
                        null : WhitespaceRegex.Replace(this.result.SimplifiedExpression, "");
                    var approx = this.result.HasFailed ? null : this.result.Approximation.ToString();

                    bool simplVisible;
                    bool approxVisible;
                    if (original == "")
                    {
                        simplVisible = false;
                        approxVisible = false;
                    }
                    else
                    {
                        // only show if there actually was a simplification
                        simplVisible = simplified != null && !original.Equals(simplified);
                        // only show if there actually was a calculation or if there was an error
                        approxVisible = approx == null || !approx.Equals(simplified);
                    }

                    var historyVisible = simplVisible || approxVisible;
                    var visibility1 = historyVisible ? Visibility.Visible : Visibility.Collapsed;
                    if (this.HistoryVisibility != visibility1)
                    {
                        this.HistoryVisibility = visibility1;
                        this.OnPropertyChanged(nameof(HistoryVisibility));
                    }
                    var visibility2 = simplVisible ? Visibility.Visible : Visibility.Collapsed;
                    if (this.SimplificationVisibility != visibility2)
                    {
                        this.SimplificationVisibility = visibility2;
                        this.OnPropertyChanged(nameof(SimplificationVisibility));
                    }
                    var visibility3 = approxVisible ? Visibility.Visible : Visibility.Collapsed;
                    if (this.ApproximationVisibility != visibility3)
                    {
                        this.ApproximationVisibility = visibility3;
                        this.OnPropertyChanged(nameof(ApproximationVisibility));
                    }
                }
            }
        }

        public bool HasFailed
        {
            get => this.result.HasFailed;
        }

        public string Approximation
        {
            get
            {
                if (this.result.HasFailed)
                {
                    return ResourceLoader.GetForCurrentView().GetString("CalculationFailedText");
                }
                return $"≈ {this.result.Approximation}";
            }
        }

        public string SimplifiedExpression
        {
            get => this.result.SimplifiedExpression == null ? "" : $"= {this.result.SimplifiedExpression}";
        }

        private void OnPropertyChanged(string name)
        {
            this.PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}
