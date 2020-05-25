using System.ComponentModel;

namespace A22abb2
{
    public sealed class CalculationViewModel : INotifyPropertyChanged
    {
        private string expression = "";

        public string Expression
        {
            get => this.expression;
            set
            {
                if (this.expression != value)
                {
                    this.expression = value;
                    this.OnPropertyChanged(nameof(Expression));
                    this.OnPropertyChanged(nameof(ExpressionText));
                    this.OnPropertyChanged(nameof(IsEmpty));

                    // TODO: do not call this on the UI thread
                    var newResult = A22abb2.Eval(this.expression.Trim());
                    if (this.Result != newResult)
                    {
                        this.Result = newResult;
                        this.OnPropertyChanged(nameof(Result));
                        this.OnPropertyChanged(nameof(ResultText));
                    }
                }
            }
        }

        public string ExpressionText
        {
            get => this.IsEmpty ? "(empty)" : this.expression;
        }

        public bool IsEmpty
        {
            get => string.IsNullOrWhiteSpace(this.expression);
        }

        public double Result { get; private set; } = double.NaN;

        public bool IsValid
        {
            get => !double.IsNaN(this.Result);
        }

        public string ResultText
        {
            get
            {
                if (this.IsEmpty)
                {
                    return "(empty)";
                }
                else if (double.IsNaN(this.Result))
                {
                    return "(error)";
                }
                else
                {
                    return $"= {this.Result}";
                }
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        private void OnPropertyChanged(string name)
        {
            this.PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}
