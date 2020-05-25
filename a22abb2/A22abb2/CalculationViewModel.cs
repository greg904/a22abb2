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

                    // TODO: do not call this on the UI thread
                    var newResult = Ffi.Eval(this.expression.Trim());
                    if (!this.result.Equals(newResult))
                    {
                        this.result = newResult;
                        this.OnPropertyChanged(nameof(HasFailed));
                        this.OnPropertyChanged(nameof(ResultValue));
                        this.OnPropertyChanged(nameof(SimplifiedExpression));
                    }
                }
            }
        }

        private bool expressionIsEmpty
        {
            get => string.IsNullOrWhiteSpace(this.expression);
        }

        public string ExpressionText
        {
            get => this.expressionIsEmpty ? "(empty)" : this.expression;
        }

        private Ffi.EvalResult result;

        public bool HasFailed
        {
            get => this.result.HasFailed;
        }

        public string ResultValue
        {
            get => this.result.HasFailed ? "" : $"≈ {this.result.ResultValue}";
        }

        public string SimplifiedExpression
        {
            get
            {
                if (this.expressionIsEmpty)
                {
                    return "(empty)";
                }
                else if (result.SimplifiedExpression == null)
                {
                    return "(error)";
                }
                else
                {
                    return $"= {this.result.SimplifiedExpression}";
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
