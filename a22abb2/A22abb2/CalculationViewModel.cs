using System;
using System.ComponentModel;
using Windows.UI.Xaml.Data;

namespace A22abb2
{
    public sealed class CalculationViewModel : INotifyPropertyChanged, ICloneable
    {
        private string expression;
        private string result;

        public string Expression {
            get => expression;
            set
            {
                if (expression != value)
                {
                    expression = value;
                    OnPropertyChanged(nameof(Expression));
                    OnPropertyChanged(nameof(Line));
                }
            }
        }

        public string Result
        {
            get => result;
            set
            {
                if (result != value)
                {
                    result = value;
                    OnPropertyChanged(nameof(Result));
                    OnPropertyChanged(nameof(Line));
                }
            }
        }

        public string Line { get => Expression + " = " + Result ?? "(error)"; }

        public CalculationViewModel(string expression, string result)
        {
            Expression = expression;
            Result = result;
        }

        public CalculationViewModel() : this(string.Empty, null)
        {
        }

        public event PropertyChangedEventHandler PropertyChanged;

        public void Reset()
        {
            Expression = string.Empty;
            Result = null;
        }

        public object Clone()
        {
            return MemberwiseClone();
        }

        private void OnPropertyChanged(string name)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }

    public sealed class PrependEqualsConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, string language)
        {
            if (targetType != typeof(string))
            {
                return null;
            }

            if (value == null)
            {
                return string.Empty;
            }

            var valString = value as string;
            if (valString == null)
            {
                return "(error)";
            }

            if (valString.Length == 0)
            {
                return "";
            }

            return $"= {valString}";
        }

        public object ConvertBack(object value, Type targetType, object parameter, string language)
        {
            throw new NotImplementedException();
        }
    }
}
