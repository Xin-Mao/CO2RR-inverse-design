import os
import re
import yaml
import numpy as np
import pandas as pd
import collections
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure, Lattice
from mendeleev import element
from collections import defaultdict
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import *


class t2mat_analyze:
    def __init__(self,
                 poscars_path,
                 training_data_path,
                 opt_steps=500,
                 input_file='input.yaml',
                 model_info='model_info.yaml',
                 info_summary=None):
        self.poscars_path = poscars_path
        self.training_data_path = training_data_path
        self.opt_steps = opt_steps
        self.settings = self.read_yaml(input_file)
        self.property_list = self.settings['property']
        self.property_list_range = self.settings['property_range']
        self.model_info_list = self.read_yaml(model_info)
        if info_summary:
            self.info_summary = info_summary
        else:
            self.info_summary = self.get_info_summary()
        self.atom_types = self.get_atom_types()

    @staticmethod
    def read_yaml(yaml_name):
        with open(yaml_name, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def get_info_summary(self, filename='info_summary'):
        info_summary = collections.defaultdict(dict)
        poscars = os.listdir(self.poscars_path)
        dict_index = {}
        for poscar in poscars:
            prop_rows = re.findall(r'\((.*?)\)', poscar)
            property_values = [
                eval(prop_row.replace('_', '.')) for prop_row in prop_rows
            ]
            # try:
            step = int(re.findall(r"step\-?\d+", poscar)[0].strip('step'))           
            dict_index[step] = dict_index.get(step, 0) + 1
            struct = Structure.from_file(f'{self.poscars_path}/{poscar}')
            formula = struct.formula.replace(' ', '')
            single_info = {
                k: v
                for k, v in zip(self.property_list, property_values)
            }
            single_info['objective_function'] = sum([
                (property_value - model_info[2]) / model_info[3]
                for property_value, model_info in zip(
                    property_values, self.model_info_list.values())
            ])
            single_info['formula'] = formula
            single_info['struct'] = struct
            info_summary[step][dict_index[step]] = single_info
            # except:
            #     print(poscar)
        info_summary = dict(sorted(info_summary.items(), key=lambda d: d[0]))
        np.save(f'{filename}.npy', info_summary)
        return info_summary

    def get_alloy_ele_count(self):
        atom_type_dict = {}
        for value in self.info_summary.values():
            for info in value.values():
                atom_type = str(
                    sorted(list(set(info['struct'].atomic_numbers))))
                atom_type_dict[atom_type] = atom_type_dict.get(atom_type,
                                                               0) + 1
        ele_dict = defaultdict(int)
        for i in list(atom_type_dict.keys()):
            ele_list = eval(i)
            for j in range(len(ele_list)):
                ele_dict[element(ele_list[j]).symbol] += 1
        ele_dict = dict(
            sorted(ele_dict.items(), key=lambda item: item[1], reverse=True))
        atom_type_dict = dict(
            sorted(atom_type_dict.items(),
                   key=lambda item: item[1],
                   reverse=True))
        return ele_dict, atom_type_dict

    def get_atom_types(self):
        atom_types = []
        for value in self.info_summary.values():
            for info in value.values():
                atom_type = info['struct'].formula.split(' ')
                atom_type = sorted(atom_type)
                atom_types.append(atom_type)
        return atom_types

    @staticmethod
    def get_ele_str(top_sp):
        ele_strs = []
        for i in list(top_sp.keys()):
            ele_str = ''
            ele_list = eval(i)
            for j in range(len(ele_list)):
                ele_str += element(ele_list[j]).symbol
            ele_strs.append(ele_str)
        return ele_strs

    @staticmethod
    def get_sp(struct, symprec=0.2, angle_tolerance=10):
        analyzer = SpacegroupAnalyzer(struct,
                                      symprec=symprec,
                                      angle_tolerance=angle_tolerance)
        return analyzer.get_space_group_number()

    def judge_if_satisfy(self,
                         single_info,
                         tolerance,
                         max_atom_nums,
                         max_composition,
                         range_toleration,
                         other_limitation=''):
        all_in_range = True
        # print('=================')
        struct = single_info['struct']
        if 'ehull' in self.property_list:
            single_info['ehull'] = single_info['ehull'] / len(struct)
        for property_name, value in single_info.items():
            if property_name in self.property_list:
                property_index = self.property_list.index(property_name)
                range_or_optimal = self.property_list_range[property_index]
                # print(range_or_optimal)
                # print(value)
                if len(range_or_optimal) == 2:
                    low, high = range_or_optimal
                    if range_toleration:
                        low -= tolerance * low
                        high += tolerance * high
                    if not (low <= value <= high):
                        all_in_range = False
                        break
                else:
                    optimal = range_or_optimal
                    if abs(value - optimal) > optimal * tolerance:
                        all_in_range = False
                        break
        # print(f'1: {all_in_range}')
        # print(len(struct))
        # print(len(struct.composition))
        if len(struct) > max_atom_nums or len(
                struct.composition) > max_composition:
            all_in_range = False
        # print(f'2: {all_in_range}')
        if other_limitation:
            if eval(other_limitation):
                all_in_range = False
        # print(f'final: {all_in_range}')
        return all_in_range

    def get_step_mean_value(self,
                            limitation_list=None,
                            tolerance=0.15,
                            max_atom_nums=100,
                            max_composition=10,
                            range_toleration=True,
                            other_limitation=None):
        info_step_dicts = defaultdict(list)
        for step in sorted(list(self.info_summary.keys())):
            info_step = list(self.info_summary[step].values())
            if 'ehull' in self.property_list:
                info_step_temp = []
                for info_dict in info_step:
                    info_dict['ehull'] = info_dict['ehull'] / len(
                        info_dict['struct'])
                    info_step_temp.append(info_dict)
                info_step = info_step_temp
            for property_name, limitation_range in zip(self.property_list,
                                                       limitation_list):
                info_step_dicts[property_name].append(
                    np.mean([
                        info_dict[property_name] for info_dict in info_step
                        if limitation_range[0] <= info_dict[property_name] <=
                        limitation_range[1]
                    ]))
            info_step_dicts['objective_function'].append(
                np.mean([
                    info_dict['objective_function'] for info_dict in info_step
                ]))
            info_step_dicts['success'].append(
                len([
                    1 for info_dict in info_step if self.judge_if_satisfy(
                        info_dict, tolerance, max_atom_nums, max_composition,
                        range_toleration, other_limitation)
                ]))
        return info_step_dicts

    def get_filtered_info(self,
                          tolerance=0.15,
                          max_atom_nums=35,
                          max_composition=5,
                          range_toleration=False,
                          other_limitation=None):
        opt_info_summary = collections.defaultdict(dict)
        opt_info_summary_tol = collections.defaultdict(dict)
        count = 0
        opt_count = []
        filtered_keys = {}
        for step in range(-1, len(self.info_summary) - 1):
            for index in range(1, len(self.info_summary[step]) + 1):
                try:
                    single_info = self.info_summary[step][index]
                    objective_function = single_info['objective_function']
                    if_satisfy = self.judge_if_satisfy(single_info, tolerance,
                                                       max_atom_nums,
                                                       max_composition,
                                                       range_toleration,
                                                       other_limitation)
                    if if_satisfy:
                        opt_info_summary_tol[step][index] = single_info
                        opt_info_summary[count] = single_info
                        filtered_keys[objective_function] = (step, index)
                        count += 1
                except:
                    print(f'step{step},index{index} error')
        for i in opt_info_summary_tol:
            opt_count.append(len(opt_info_summary_tol[i]))
        return opt_info_summary, opt_info_summary_tol, opt_count, filtered_keys

    @staticmethod
    def get_alloy_ele_count_top(opt_info_summary, top=200):
        atom_type_dict = {}
        for info in list(opt_info_summary.values())[:top]:
            atom_type = str(sorted(list(set(info['struct'].atomic_numbers))))
            atom_type_dict[atom_type] = atom_type_dict.get(atom_type, 0) + 1
        ele_dict = defaultdict(int)
        for i in list(atom_type_dict.keys()):
            ele_list = eval(i)
            for j in range(len(ele_list)):
                ele_dict[element(ele_list[j]).symbol] += 1
        ele_dict = dict(
            sorted(ele_dict.items(), key=lambda item: item[1], reverse=True))
        atom_type_dict = dict(
            sorted(atom_type_dict.items(),
                   key=lambda item: item[1],
                   reverse=True))
        return ele_dict, atom_type_dict

    def get_best_res(self,
                     filtered_keys,
                     top_value,
                     screen_sp=False,
                     write_vasp=False):
        best_values = sorted(list(filtered_keys.keys()))
        best_keys = [
            filtered_keys[best_value] for best_value in best_values[:top_value]
        ]
        best_res = defaultdict(dict)
        structs = []
        for index, best_key in enumerate(best_keys):
            single_info = self.info_summary[best_key[0]][best_key[1]]
            struct = single_info['struct']
            structs.append(struct)
            best_res[struct.formula.replace(' ', '') + '_' +
                     str(index)] = single_info
        best_res_df = pd.DataFrame(best_res).T
        best_res_df.to_csv('best_res_df.csv')
        if not os.path.exists(f'best_struct{top_value}'):
            os.mkdir(f'best_struct{top_value}')
        if screen_sp:
            for index, best_struct in enumerate(structs):
                if self.get_sp(best_struct) != 1:
                    self.write_poscar(best_struct, top_value, index)
                    if write_vasp:
                        self.write_vasp_input(best_struct, 'best_struct_vasp',
                                              index)
        else:
            for index, best_struct in enumerate(structs):
                self.write_poscar(best_struct, top_value, index)
                if write_vasp:
                    self.write_vasp_input(best_struct, 'best_struct_vasp',
                                          index)
        return best_res_df, structs

    @staticmethod
    def write_poscar(struct, expe_num, index):
        poscar = Poscar(struct)
        formu = struct.formula.replace(' ', '') + '_' + str(index)
        poscar.write_file(f'best_struct{expe_num}/POSCAR_{formu}')

    @staticmethod
    def write_vasp_input(struct, file_path, index, vasp_input_set=MPRelaxSet):
        relax_set = vasp_input_set(structure=struct)
        formu = struct.formula.replace(' ', '') + '_' + str(index)
        output_path = file_path + '/' + formu
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        relax_set.write_input(output_dir=output_path, potcar_spec=False)

    @staticmethod
    def get_structure(struct):
        lattice = Lattice.from_parameters(struct[-2][0], struct[-2][1],
                                          struct[-2][2], struct[-1][0],
                                          struct[-1][1], struct[-1][2])
        structure = Structure.from_spacegroup(sg='P1',
                                              lattice=lattice,
                                              species=struct[0],
                                              coords=struct[1])
        return structure

    @staticmethod
    def get_com(com_str):
        ratio = re.findall(r"\d+", com_str)
        ele = re.findall(r"\D+", com_str)
        comp = [i + j for i, j in zip(ele, ratio)]
        comp = sorted(comp)
        return comp

    def get_training_info(self):
        train_df = pd.read_csv(f'{self.training_data_path}train.csv',
                               index_col=0)
        val_df = pd.read_csv(f'{self.training_data_path}val.csv', index_col=0)
        train_com = list(train_df['material_id'])
        val_com = list(val_df['material_id'])
        train_com = [self.get_com(com_str) for com_str in train_com]
        val_com = [self.get_com(com_str) for com_str in val_com]
        return train_com, val_com, train_df, val_df

    def cal_gen_rate(self, atom_types, train_com, val_com, structures,
                     train_df, val_df):
        count = 0
        count_index = []
        for index, atom_type in enumerate(atom_types):
            if atom_type in train_com or atom_type in val_com:
                count += 1
                count_index.append(index)
        matcher = StructureMatcher(stol=0.5,
                                   angle_tol=10,
                                   ltol=0.3,
                                   attempt_supercell=True,
                                   primitive_cell=False)
        rms_dists = []
        for index in count_index:
            opt_structure = structures[index]
            try:
                train_index = train_com.index(atom_types[index])
                struct_cif = CifParser.from_string(
                    list(train_df['cif'])[train_index])
            except:
                val_index = val_com.index(atom_types[index])
                struct_cif = CifParser.from_string(
                    list(val_df['cif'])[val_index])
            train_val_structure = struct_cif.get_structures()[0]
            rms_dist = matcher.get_rms_dist(opt_structure, train_val_structure)
            rms_dists.append(rms_dist)
        rms_dists_pure = [i for i in rms_dists if i != None]
        return count_index, rms_dists, rms_dists_pure, count / len(
            atom_types), len(rms_dists_pure) / len(atom_types)

    def cal_gen_rate_relative(self,
                              atom_types1,
                              atom_types2,
                              structs1,
                              structs2,
                              stol=0.5,
                              angle_tol=10,
                              ltol=0.3,
                              attempt_supercell=True,
                              primitive_cell=False):
        count = 0
        count_index1 = []
        count_index2 = []
        for index, atom_type in enumerate(atom_types1):
            if atom_type in atom_types2:
                count += 1
                count_index1.append(index)
                count_index2.append(atom_types2.index(atom_type))
        matcher = StructureMatcher(stol=stol,
                                   angle_tol=angle_tol,
                                   ltol=ltol,
                                   attempt_supercell=attempt_supercell,
                                   primitive_cell=primitive_cell)
        rms_dists = []
        for index1, index2 in zip(count_index1, count_index2):
            opt_structure1 = structs1[index1]
            opt_structure2 = structs2[index2]
            rms_dist = matcher.get_rms_dist(opt_structure1, opt_structure2)
            rms_dists.append(rms_dist)
        rms_dists_pure = [i for i in rms_dists if i != None]
        # print(atom_types1, atom_types2)
        atom_types_len = min(len(atom_types1), len(atom_types2))
        # print(atom_types_len)
        return count_index1, count_index2, rms_dists, rms_dists_pure, count / len(
            self.atom_types), len(rms_dists_pure) / atom_types_len

    def get_gen_rate(self, train_com, val_com, train_df, val_df):
        com_same_rates = []
        struct_same_rates = []
        for value in self.info_summary.values():
            atom_types = []
            structs = []
            for info in value.values():
                structs.append(info['struct'])
                atom_type = info['struct'].formula.split(' ')
                atom_type = sorted(atom_type)
                atom_types.append(atom_type)
            _, _, _, com_same_rate, struct_same_rate = self.cal_gen_rate(
                atom_types, train_com, val_com, structs, train_df, val_df)
            com_same_rates.append(com_same_rate)
            struct_same_rates.append(struct_same_rate)
        return com_same_rates, struct_same_rates

    def get_similiar_rates_relative(self,
                                    optimal_structs=False,
                                    tolerance=0.15,
                                    max_atom_nums=35,
                                    max_composition=5,
                                    range_toleration=False,
                                    other_limitation=None):
        com_same_rates_relative = []
        struct_same_rates_relative = []
        for key, value in self.info_summary.items():
            if key > -1:
                atom_types1 = []
                structs1 = []
                atom_types2 = []
                structs2 = []
                for info in value.values():
                    if optimal_structs:
                        if_append = self.judge_if_satisfy(
                            info, tolerance, max_atom_nums, max_composition,
                            range_toleration, other_limitation)
                    else:
                        if_append = True
                    if if_append:
                        structs1.append(info['struct'])
                        atom_type1 = info['struct'].formula.split(' ')
                        atom_type1 = sorted(atom_type1)
                        atom_types1.append(atom_type1)
                for info in self.info_summary[key - 1].values():
                    if optimal_structs:
                        if_append = self.judge_if_satisfy(
                            info, tolerance, max_atom_nums, max_composition,
                            range_toleration, other_limitation)
                    else:
                        if_append = True
                    if if_append:
                        structs2.append(info['struct'])
                        atom_type2 = info['struct'].formula.split(' ')
                        atom_type2 = sorted(atom_type2)
                        atom_types2.append(atom_type2)
                _, _, _, _, com_same_rate, struct_same_rate = self.cal_gen_rate_relative(
                    atom_types1, atom_types2, structs1, structs2)
                com_same_rates_relative.append(com_same_rate)
                struct_same_rates_relative.append(struct_same_rate)
        return com_same_rates_relative, struct_same_rates_relative

    def auto_analyze(self,
                     tolerance=0.15,
                     top=200,
                     max_atom_nums=35,
                     max_composition=5,
                     range_toleration=False,
                     other_limitation=None,
                     get_rate=True,
                     top_value=200,
                     limitation_list=None,
                     optimal_structs=False):
        self.info_step_dicts = self.get_step_mean_value(
            limitation_list, tolerance, max_atom_nums, max_composition,
            range_toleration, other_limitation)
        np.save('info_step_dicts.npy', self.info_step_dicts)
        self.ele_dict, self.atom_type_dict = self.get_alloy_ele_count()
        np.save('ele_dict.npy', self.ele_dict)
        np.save('atom_type_dict.npy', self.atom_type_dict)
        self.opt_info_summary, self.opt_info_summary_tol, self.opt_count, self.filtered_keys = self.get_filtered_info(
            tolerance, max_atom_nums, max_composition, range_toleration,
            other_limitation)
        np.save('filtered_keys.npy', self.filtered_keys)
        self.best_res_df, self.best_structs = self.get_best_res(self.filtered_keys,
                                                      top_value)
        np.save('opt_info_summary.npy', self.opt_info_summary)
        np.save('opt_info_summary_tol.npy', self.opt_info_summary_tol)
        np.save('opt_count.npy', self.opt_count)
        self.ele_dict_filter, self.atom_type_dict_filter = self.get_alloy_ele_count(
            self.opt_info_summary_tol)
        np.save('ele_dict_filter.npy', self.ele_dict_filter)
        np.save('atom_type_dict_filter.npy', self.atom_type_dict_filter)
        self.ele_dict_top, self.atom_type_dict_top = self.get_alloy_ele_count_top(
            dict(
                sorted(self.opt_info_summary.items(),
                       key=lambda x: x[1]['objective_function'])), top)
        np.save('ele_dict_top.npy', self.ele_dict_top)
        np.save('atom_type_dict_top.npy', self.atom_type_dict_top)
        if get_rate:
            self.train_com, self.val_com, self.train_df, self.val_df = self.get_training_info(
            )
            self.com_same_rates, self.struct_same_rates = self.get_gen_rate(
                self.train_com, self.val_com, self.train_df, self.val_df)
            self.com_same_rates_relative, self.struct_same_rates_relative = self.get_similiar_rates_relative(
                optimal_structs=optimal_structs,
                tolerance=tolerance,
                max_atom_nums=max_atom_nums,
                max_composition=max_composition,
                range_toleration=range_toleration,
                other_limitation=other_limitation)
            np.save('com_same_rates.npy', self.com_same_rates)
            np.save('struct_same_rates.npy', self.struct_same_rates)
            np.save('com_same_rates_relative.npy',
                    self.com_same_rates_relative)
            np.save('struct_same_rates_relative.npy',
                    self.struct_same_rates_relative)
