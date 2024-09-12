import re
from sympy import sympify, N
from sympy.printing import latex
import matplotlib.pyplot as plt
import pandas as pd


class parse_gplearn_formula:
    def __init__(self, user_define_func=None):
        self.operator_dict = {
            'div': '/',
            'mul': '*',
            'sqrt': 'sqrt',
            'add': '+',
            'sub': '-',
            'pow': '^',
            'log': 'log',
            'exp': 'exp',
            'cbrt': 'cbrt',
            'abs': 'abs'
        }
        if user_define_func:
            for func in user_define_func:
                self.operator_dict[func] = func

    def parse(self, formula):
        formula = formula.lstrip()
        operator = formula.split('(')[0]
        if operator in [
                'div', 'mul', 'add', 'sub', 'pow', 'log', 'exp', 'cbrt',
                'sqrt', 'abs'
        ]:
            return self.parse_operator(formula, operator)
        else:
            return self.parse_variable(formula)

    def parse_operator(self, formula, operator):
        operator_str = self.operator_dict[operator]
        split_len = len(operator) + 1
        formula = formula[split_len:].lstrip()
        if operator in ['log', 'sqrt', 'exp', 'cbrt', 'abs']:
            inside, formula = self.parse(formula)
            formula = formula.lstrip()
            return '{}({})'.format(operator_str, inside), formula[1:]
        else:
            numerator, formula = self.parse(formula)
            formula = formula.lstrip()
            formula = formula[1:].lstrip()
            denominator, formula = self.parse(formula)
            formula = formula.lstrip()
            return '({} {} {})'.format(numerator, operator_str,
                                       denominator), formula[1:]

    def parse_variable(self, formula):
        match = re.match(r'([A-Za-z0-9_.]+)(.*)', formula.lstrip())
        return match.group(1), match.group(2)

    def convert_formula(self, formula):
        result, _ = self.parse(formula)
        return result

    def get_latex(self, formula, decimal=4, already_convert=False):
        if already_convert:
            formula_str = formula
        else:
            formula_str = self.convert_formula(formula)
        formula = sympify(formula_str)
        formula = N(formula, decimal)
        return latex(formula)

    def visualize_formula(self,
                          formula,
                          decimal=4,
                          size=100,
                          format='pdf',
                          dpi=1200,
                          save_fig=True,
                          save_name='formula',
                          use_latex=False,
                          font=None,
                          already_convert=False):
        latex_code = self.get_latex(formula,
                                    decimal,
                                    already_convert=already_convert)
        if use_latex:
            plt.rcParams['text.usetex'] = True
        if font:
            plt.rcParams['font.family'] = font
        plt.axis('off')
        plt.text(0.5,
                 0.5,
                 f'${latex_code}$',
                 size=size,
                 ha='center',
                 va='center')
        if save_fig:
            plt.savefig(f'{save_name}.{format}', dpi=dpi, bbox_inches='tight')
        plt.show()


class parse_sisso_formula(parse_gplearn_formula):
    def __init__(self, user_define_func=None, file_path=''):
        super().__init__(user_define_func)
        self.file_path = file_path

    def read_sisso_out(self, file_path='SISSO.out', decimal=4, save_csv=True):
        with open(file_path, 'r') as f:
            content = f.read()
            parts = content.split('@@@descriptor:')
            descriptor_info = {}
            for i, part in enumerate(parts[1:], start=1):
                descriptor = ''
                descriptors = re.findall(r'\[([^\]]+)\]', part)
                coefficients = re.findall(r'coefficients_001:\s+([^\n]+)',
                                          part)[0].split('    ')
                coefficients = [
                    float(coefficient) for coefficient in coefficients
                ]
                intercept = re.search(r'Intercept_001:\s+([^\s]+)',
                                      part).group(1)
                rmse, maxae = re.search(
                    r'RMSE,MaxAE_001:\s+([^\s]+)\s+([^\s]+)', part).groups()
                for descrip, coefficient in zip(descriptors, coefficients):
                    coefficient = str(coefficient)
                    descriptor += f'{descrip}*{coefficient}+'
                descriptor += f'{str(float(intercept))}'
                descriptor_latex = self.get_latex(descriptor,
                                                  decimal=decimal,
                                                  already_convert=True)
                descriptor_info[f'{i}D descriptor'] = {
                    'descriptor': descriptor,
                    'coefficients': coefficients,
                    'intercepts': float(intercept),
                    'rmse': float(rmse),
                    'maxae': float(maxae),
                    'descriptor_latex': descriptor_latex
                }
            if save_csv:
                df = pd.DataFrame(descriptor_info)
                df.to_csv('descriptor_info.csv')
        return descriptor_info

    def visualize_descriptor(self,
                             dimension=None,
                             decimal=4,
                             size=100,
                             format='pdf',
                             dpi=1200,
                             save_fig=True,
                             use_latex=False,
                             font=None):
        file_path = self.file_path + 'SISSO.out'
        descriptor_info = self.read_sisso_out(file_path)
        if not dimension:
            for key, value in descriptor_info.items():
                print(key)
                self.visualize_formula(value['descriptor'],
                                       save_name=key,
                                       save_fig=save_fig,
                                       already_convert=True,
                                       decimal=decimal,
                                       size=size,
                                       format=format,
                                       dpi=dpi,
                                       use_latex=use_latex,
                                       font=font)
        else:
            key = f'{dimension}D descriptor'
            self.visualize_formula(descriptor_info[key]['descriptor'],
                                   save_name=key,
                                   save_fig=save_fig,
                                   already_convert=True,
                                   decimal=decimal,
                                   size=size,
                                   format=format,
                                   dpi=dpi,
                                   use_latex=use_latex,
                                   font=font)
