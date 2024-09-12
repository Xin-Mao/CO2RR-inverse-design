import os
import re
import ase
import json
from ase import io
from ase.io import cif
import numpy as np
from ase import Atoms, Atom
import torch
import random
import pandas as pd
import collections
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor
from mendeleev import element
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib.pyplot import MultipleLocator
from t2mat_util.analyze_utilities import t2mat_analyze


class t2mat_visualize(t2mat_analyze):
    def __init__(self,
                 ana_instance=None,
                 if_analyze=False,
                 poscars_path=None,
                 property_list=None,
                 optimal_range=None,
                 info_summary=None):
        if if_analyze and ana_instance is not None:
            for key, value in vars(ana_instance).items():
                setattr(self, key, value)
        else:
            super().__init__(poscars_path, property_list, optimal_range,
                             info_summary)
            if info_summary:
                self.info_summary = info_summary
            else:
                self.info_summary = self.get_info_summary()
            self.ele_dict, self.atom_type_dict = self.get_alloy_ele_count()
        self.if_analyze = if_analyze

    @staticmethod
    def set_figure_property(font_size=32,
                            font_family='Helvetica',
                            font='Liberation Sans',
                            bwith=2,
                            tick_width=3,
                            tick_length=10,
                            if_return=False):
        plt.rcParams['font.size'] = font_size
        plt.rcParams['font.family'] = font_family
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        plt.tick_params(which='major', width=tick_width, length=tick_length)
        if if_return:
            return ax

    def draw_alloy(self, atom_type_dict, name, num, fig_size=(35, 10)):
        top_sp = dict(tuple(atom_type_dict.items())[:num])
        plt.figure(figsize=fig_size)
        up = round(max(list(top_sp.values())) + 2, 0)
        down = round(min(list(top_sp.values())) - 1, 0)
        norm = plt.Normalize(down, up)
        top_sp_keys = self.get_ele_str(top_sp)
        norm_values = norm(list(top_sp.values()))
        map_vir = cm.get_cmap(name='YlGnBu')
        colors = map_vir(norm_values)
        plt.bar(top_sp_keys,
                list(top_sp.values()),
                width=0.6,
                alpha=0.8,
                color=colors,
                edgecolor='black',
                linewidth=3)
        self.set_figure_property()
        plt.ylabel("Numbers")
        plt.savefig(name, bbox_inches='tight', dpi=1200)

    def draw_mendeleev(self, plot_data, down, up, filename, fig_size=(20, 10)):
        # 元素周期表中cell的设置
        # cell的大小
        cell_length = 1
        # 各个cell的间隔
        cell_gap = 0.1
        # cell边框的粗细
        cell_edge_width = 0.5

        # 获取各个元素的原子序数、周期数（行数）、族数（列数）以及绘制数据（没有的设置为0）
        elements = []
        for i in range(1, 119):
            ele = element(i)
            ele_group, ele_period = ele.group_id, ele.period

            # 将La系元素设置到第8行
            if 57 <= i <= 71:
                ele_group = i - 57 + 3
                ele_period = 8
            # 将Ac系元素设置到第9行
            if 89 <= i <= 103:
                ele_group = i - 89 + 3
                ele_period = 9
            elements.append([
                i, ele.symbol, ele_group, ele_period,
                plot_data.setdefault(ele.symbol, 'None')
            ])

        # 设置La和Ac系的注解标签
        elements.append([None, 'LA', 3, 6, None])
        elements.append([None, 'AC', 3, 7, None])
        elements.append([None, 'LA', 2, 8, None])
        elements.append([None, 'AC', 2, 9, None])

        # 新建Matplotlib绘图窗口
        plt.figure(figsize=fig_size)
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.sans-serif'] = ['Liberation Sans']
        # x、y轴的范围
        xy_length = (20, 11)

        # 获取YlOrRd颜色条
        my_cmap = cm.get_cmap('YlGnBu')
        # 将plot_data数据映射为颜色，根据实际情况调整
        norm = mpl.colors.Normalize(down, up)
        # 设置超出颜色条下界限的颜色（None为不设置，即白色）
        my_cmap.set_under('None')
        # 关联颜色条和映射
        cmmapable = cm.ScalarMappable(norm, my_cmap)
        # 绘制颜色条
        cb = plt.colorbar(cmmapable, drawedges=False)
        tick_locator = ticker.MaxNLocator(nbins=10)
        cb.locator = tick_locator
        cb.update_ticks()
        # 绘制元素周期表的cell，并填充属性和颜色
        for e in elements:
            ele_number, ele_symbol, ele_group, ele_period, ele_count = e
            # print(ele_number, ele_symbol, ele_group, ele_period, ele_count)

            if ele_group is None:
                continue
            # x, y定位cell的位置
            x = (cell_length + cell_gap) * (ele_group - 1)
            y = xy_length[1] - ((cell_length + cell_gap) * ele_period)

            # 增加 La, Ac 系元素距离元素周期表的距离
            if ele_period >= 8:
                y -= cell_length * 0.5

            # cell中原子序数部位None时绘制cell边框并填充热力颜色
            # 即不绘制La、Ac系注解标签地边框以及颜色填充
            if ele_number:
                if ele_count == 'None':
                    fill_color = (1.0, 1.0, 1.0, 1.0)
                else:
                    fill_color = my_cmap(norm(ele_count))
                rect = patches.Rectangle(xy=(x, y),
                                         width=cell_length,
                                         height=cell_length,
                                         linewidth=cell_edge_width,
                                         edgecolor='k',
                                         facecolor=fill_color)
                plt.gca().add_patch(rect)

            # 在cell中添加原子序数属性
            plt.text(
                x + 0.04,
                y + 0.8,
                ele_number,
                va='center',
                ha='left',
                #  fontdict={'size': 14, 'color': 'black', 'family': 'Helvetica'})
                fontdict={
                    'size': 14,
                    'color': 'black'
                })
            # 在cell中添加元素符号
            plt.text(
                x + 0.5,
                y + 0.5,
                ele_symbol,
                va='center',
                ha='center',
                #  fontdict={'size': 14, 'color': 'black', 'family': 'Helvetica', 'weight': 'bold'})
                fontdict={
                    'size': 14,
                    'color': 'black',
                    'weight': 'bold'
                })
            # 在cell中添加热力值
            plt.text(
                x + 0.5,
                y + 0.12,
                ele_count,
                va='center',
                ha='center',
                #  fontdict={'size': 14, 'color': 'black', 'family': 'Helvetica'})
                fontdict={
                    'size': 14,
                    'color': 'black'
                })
        # x, y 轴设置等比例（1:1）（使cell看起来是正方形）
        # plt.axis('equal')
        # 关闭坐标轴
        plt.axis('off')
        # 裁剪空白边缘
        plt.tight_layout()
        # 设置x, y轴的范围
        plt.ylim(0, xy_length[1])
        plt.xlim(0, xy_length[0])

        # 将图保存为*.svg矢量格式
        plt.savefig(filename, bbox_inches='tight', dpi=1200)
        # 显示绘图窗口
        plt.show()

    def draw_property_distribution(self,
                                   property_name,
                                   ylabel,
                                   save_name,
                                   fig_size=(18, 24)):
        energies = []
        for step in range(-1, len(self.info_summary) - 1):
            for index in range(1, len(self.info_summary[step]) + 1):
                ene = self.info_summary[step][index][property_name]
                energies.append(ene)
        plt.figure(figsize=fig_size)
        sns.violinplot(y=energies, linewidth=2)
        self.set_figure_property()
        plt.ylabel(ylabel)
        plt.savefig(save_name, bbox_inches='tight', dpi=1200)

    def draw_bsa_step(self,
                      property_list,
                      name,
                      ylabel,
                      scale=1,
                      markersize=20,
                      up=None,
                      down=None,
                      all_results=False,
                      fig_size=(16, 14)):
        plt.figure(figsize=fig_size)
        steps = np.array(list(range(len(property_list)))) * scale
        plt.plot(steps,
                 property_list,
                 marker='o',
                 linewidth=3.0,
                 linestyle='-',
                 markersize=markersize,
                 markerfacecolor='silver',
                 markeredgewidth=3,
                 markeredgecolor='k')
        if all_results:
            plt.fill_between(steps,
                             property_list - down,
                             property_list + up,
                             alpha=0.2)
        self.set_figure_property()
        plt.ylabel(ylabel)
        plt.xlabel("BSA steps")
        plt.savefig(name, bbox_inches='tight', dpi=1200)

    def draw_bsa_best(self,
                      property_name,
                      name,
                      ylabel,
                      limitation,
                      scale=1,
                      fig_size=(16, 14)):
        best_values = []
        for value in self.info_summary.values():
            values_temp = []
            for info in value.values():
                if limitation[0] <= info[property_name] <= limitation[1]:
                    values_temp.append(info[property_name])
            best_values.append(min(values_temp))
        best_values_best = [
            min(best_values[:i + 1]) for i in range(len(best_values))
        ]
        plt.figure(figsize=fig_size)
        self.set_figure_property()
        steps = np.array(list(range(len(best_values_best)))) * scale
        plt.plot(steps, best_values_best, linewidth=3.0, linestyle='-')
        change_index = [
            i for i in range(len(best_values_best))
            if best_values_best[i - 1] != best_values_best[i]
        ]
        change_value = [best_values_best[i] for i in change_index]
        plt.scatter(change_index,
                    change_value,
                    marker='o',
                    edgecolors='black',
                    c='grey',
                    s=400,
                    linewidths=3,
                    zorder=1000)
        plt.ylabel(ylabel)
        plt.xlabel("BSA steps")
        plt.savefig(name, bbox_inches='tight', dpi=1200)

    def draw_bsa_step_avg(self,
                          list_avg,
                          list_all,
                          name,
                          ylabel,
                          scale=1,
                          markersize=20,
                          up=None,
                          down=None,
                          fig_size=(16, 15)):
        plt.figure(figsize=fig_size)
        steps = np.array(list(range(len(list_avg)))) * scale
        plt.plot(steps,
                 list_avg,
                 marker='o',
                 linewidth=3.0,
                 linestyle='-',
                 markersize=markersize,
                 markerfacecolor='silver',
                 markeredgewidth=3,
                 markeredgecolor='k')
        down = np.array(
            [np.sum(down[i * 10:(i + 1) * 10]) / 1000 for i in range(50)])
        up = np.array(
            [np.sum(up[i * 10:(i + 1) * 10]) / 1000 for i in range(50)])
        plt.fill_between(steps, (list_avg - down), (list_avg + up), alpha=0.2)
        self.set_figure_property()
        plt.ylim([0.15, 0.52])
        plt.ylabel(ylabel)
        plt.xlabel("BSA steps")
        plt.savefig(name, bbox_inches='tight', dpi=1200)

    def draw_pareto_front(self,
                          struct_ene_map_diff,
                          name,
                          xlabel,
                          ylabel,
                          fig_size=(16, 12)):
        plt.figure(fig_size)
        co_ene = struct_ene_map_diff['energy_co']
        h_ene = struct_ene_map_diff['energy_h']
        plt.scatter(co_ene,
                    h_ene,
                    marker='o',
                    alpha=0.5,
                    s=150,
                    c=struct_ene_map_diff['formation_ene'])
        plt.colorbar()
        # plt.plot(noise_value, list2, marker='o',  linewidth=2.0, linestyle='-', markersize=10, markerfacecolor='w',
        #     markeredgewidth=2, markeredgecolor='k')
        # plt.legend(labels=['Mean MEGNet predicted formation energies','Mean Voronoi polyhedra ratios'],loc='best')
        ax = self.set_figure_property(if_return=True)
        x_major_locator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_major_locator)
        plt.tick_params(which='major', width=3, length=10)
        plt.ylabel(xlabel)
        plt.xlabel(xlabel)
        x_tick_spacing = 0.05
        plt.xticks(np.arange(0, 0.25, x_tick_spacing))
        plt.xlim((-0.015, 0.25))
        plt.ylim((1.1, -0.1))
        plt.savefig(name, bbox_inches='tight', dpi=1200)

    def auto_visualize(self,
                       tolerance=0.15,
                       top=200,
                       max_atom_nums=35,
                       max_composition=5,
                       range_toleration=False,
                       other_limitation=None,
                       get_rate=True,
                       optimal_structs=False,
                       draw_alloy_count=10,
                       average_step=10,
                       limitation_list=None):
        if not self.if_analyze:
            self.auto_analyze(tolerance=tolerance,
                              top=top,
                              max_atom_nums=max_atom_nums,
                              max_composition=max_composition,
                              range_toleration=range_toleration,
                              other_limitation=other_limitation,
                              get_rate=get_rate,
                              limitation_list=limitation_list,
                              optimal_structs=optimal_structs)
        for index, property_name in enumerate(self.property_list):
            self.draw_bsa_best(property_name, f'{property_name}.pdf',
                               property_name, limitation_list[index])
            self.draw_bsa_step(self.info_step_dicts[property_name],
                               f'{property_name}.pdf', property_name)
            average_step_prop = [
                np.mean(self.info_step_dicts[property_name][average_step *
                                                            i:average_step *
                                                            (i + 1)])
                for i in range(
                    int(
                        len(self.info_step_dicts[property_name]) /
                        average_step))
            ]
            self.draw_bsa_step(average_step_prop,
                               f'{property_name}_{average_step}.pdf',
                               f'{property_name}_{average_step}')
        self.draw_bsa_best('objective_function', 'objective_function.pdf',
                           'objective_function', [-100, 100])
        self.draw_bsa_step(self.info_step_dicts['objective_function'],
                           'objective_function.pdf', 'objective_function')
        average_step_obj = [
            np.mean(self.info_step_dicts['objective_function'][average_step *
                                                               i:average_step *
                                                               (i + 1)])
            for i in range(
                int(
                    len(self.info_step_dicts['objective_function']) /
                    average_step))
        ]
        self.draw_bsa_step(average_step_obj,
                           f'objective_function_{average_step}.pdf',
                           f'objective_function_{average_step}')
        self.draw_bsa_step(self.opt_count, 'opt_count_step.pdf',
                           'Number of structures satisfying requirements')
        average_step_opt_count = [
            np.mean(self.opt_count[average_step * i:average_step * (i + 1)])
            for i in range(int(len(self.opt_count) / average_step))
        ]
        self.draw_bsa_step(average_step_opt_count,
                           f'opt_count_step_{average_step}.pdf',
                           'Number of structures satisfying requirements')
        self.draw_bsa_step(self.struct_same_rates_relative,
                           'Structure similar rate.pdf',
                           'Structure similar rate')
        average_step_same_rates = [
            np.mean(self.struct_same_rates_relative[average_step *
                                                    i:average_step * (i + 1)])
            for i in range(
                int(len(self.struct_same_rates_relative) / average_step))
        ]
        self.draw_bsa_step(average_step_same_rates,
                           f'Structure similar rate {average_step}.pdf',
                           'Structure similar rate')
        self.draw_alloy(self.atom_type_dict, 'alloy_counts.pdf',
                        draw_alloy_count)
        self.draw_mendeleev(
            self.ele_dict, 0,
            max(list([i for i in self.ele_dict.values() if isinstance(i, int)
                      ])) * 1.05, 'mendeleev_distribution.pdf')
        self.draw_alloy(self.atom_type_dict_filter,
                        'alloy_counts_filtered.pdf', draw_alloy_count)
        self.draw_mendeleev(
            self.ele_dict_filter, 0,
            max(
                list([
                    i for i in self.ele_dict_filter.values()
                    if isinstance(i, int)
                ])) * 1.05, 'mendeleev_distribution_filtered.pdf')
