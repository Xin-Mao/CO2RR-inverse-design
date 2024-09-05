from numpy import *
from predict_energy import *
import warnings
import torch

warnings.filterwarnings('ignore')


class Bird_swarm_opt:
    def __init__(self,
                 opt_matrix,
                 cdvae_model,
                 ld_kwargs,
                 ck_path,
                 min_value=-6,
                 max_value=6,
                 save_traj=False,
                 traj_path='',
                 interval=100):
        self.opt_matrix = opt_matrix
        self.cdvae_model = cdvae_model
        self.ld_kwargs = ld_kwargs
        self.save_traj = save_traj
        self.traj_path = traj_path
        self.pBounds = {}
        self.kwargs = {}
        self.step = 0
        self.min_value = min_value
        self.max_value = max_value
        self.dim, self.lb, self.ub = self.get_lb_ub_dim()
        self.DimeNet = load_dimenet(ck_path)
        # self.DimeNet.freeze()
        self.interval = interval
        self.all_crystals = []

    def get_lb_ub_dim(self):
        dim = self.opt_matrix.size(1)
        lb = [self.min_value] * dim
        ub = [self.max_value] * dim
        lb = expand_dims(lb, axis=0)
        ub = expand_dims(ub, axis=0)
        return torch.tensor(dim), torch.tensor(lb), torch.tensor(ub)

    def predict_absorption_energy(self, z, index, M):
        samples = self.cdvae_model.langevin_dynamics(z, self.ld_kwargs)
        if index % self.interval == 0 or index == (M - 1):
            self.all_crystals.append(samples)
        structures = get_structures(samples['frac_coords'],
                                    samples['atom_types'],
                                    samples['num_atoms'], samples['lengths'],
                                    samples['angles'])
        energies = torch.tensor([
            abs(
                predict_ene(structure,
                            self.DimeNet,
                            write_poscar=True,
                            step=index,
                            file_path='opt_poscars') + 0.67)**0.25
            for structure in structures.values()
        ])
        return torch.tensor(array(energies).reshape(len(energies), 1)).cuda()

    def randiTabu(self, minm, maxm, tabu, dim):
        value = ones((dim, 1)) * maxm * 2
        num = 1
        while (num <= dim):
            temp = random.randint(minm, maxm)
            findi = [
                index for (index, values) in enumerate(value) if values != temp
            ]
            if (len(findi) == dim and temp != tabu):
                value[0][num - 1] = temp
                num += 1
        return value

    def Bounds(self, s, lb, ub):
        temp = s
        I = [
            index for (index, values) in enumerate(temp)
            if values < lb[0][index]
        ]
        for indexlb in I:
            temp[indexlb] = lb[0][indexlb]
        J = [
            index for (index, values) in enumerate(temp)
            if values > ub[0][index]
        ]
        for indexub in J:
            temp[indexub] = ub[0][indexub]
        return s

    def search(self,
               M=100,
               FQ=10,
               c1=0.15,
               c2=0.15,
               a1=1,
               a2=1,
               conti=False,
               conti_iteration=None):
        #############################################################################
        #     Initialization
        if conti:
            x = torch.load("x10.pt")
            pop = x.size(0)
            fit = torch.load("fit10.pt")
            pFit = torch.load("pFit10.pt")
            pX = torch.load("pX10.pt")
            fMin = torch.load("fMin10.pt")
            bestIndex = torch.load("bestIndex10.pt")
            bestX = torch.load("bestX10.pt")
            b2 = torch.load("b210.pt")
        else:
            x = self.opt_matrix
            x.cuda()
            pop = self.opt_matrix.size(0)
            fit = self.predict_absorption_energy(x, -1, M)
            fit.cuda()
            pFit = fit.clone()
            pFit.cuda()
            pX = x.clone()
            pX.cuda()
            fMin = float(min(pFit))
            # print(fMin)
            bestIndex = torch.argmin(pFit)
            bestX = pX[bestIndex, :]
            # print(bestX)
            b2 = torch.zeros([M, 1])
            b2[0] = fMin
        #     Start the iteration.
        if conti_iteration:
            inter_start = conti_iteration
        else:
            inter_start = 0
        for index in range(inter_start, M):
            print(f'Iteration {index}')
            prob = torch.rand(pop, 1) * 0.2 + 0.8
            if (mod(index, FQ) != 0):
                ###############################################################################
                #     Birds forage for food or keep vigilance
                sumPfit = pFit.sum().cuda()
                meanP = torch.mean(pX).cuda()
                realmin = torch.tensor(finfo(float).tiny).cuda()
                for i in range(0, pop):
                    if torch.rand(1).cuda() < float(prob[i]):
                        x[i, :] = x[i, :] + c1 * torch.rand(1).cuda() * (
                            bestX - x[i, :]) + c2 * torch.rand(1).cuda() * (
                                pX[i, :] - x[i, :])
                    else:
                        person = int(self.randiTabu(1, pop, i, 1)[0])
                        x[i, :] = x[i, :] + torch.rand(1).cuda() * (
                            meanP - x[i, :]) * a1 * torch.exp(
                                -pFit[i] / (sumPfit + realmin) * pop) + a2 * (
                                    torch.rand(1).cuda() * 2 - 1
                                ) * (pX[person, :] - x[i, :]) * torch.exp(
                                    -(pFit[person] - pFit[i]) /
                                    (abs(pFit[person] - pFit[i]) + realmin) *
                                    pFit[person] / (sumPfit + realmin) * pop)
    #                 print(x[i,:])
                    x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                    #                 print(x[i,:])
                fit = self.predict_absorption_energy(x, index, M)
    ###############################################################################
            else:
                FL = torch.rand(pop, 1) * 0.4 + 0.5
                ###############################################################################
                #     Divide the bird swarm into two parts: producers and scroungers
                minIndex = torch.argmin(pFit)
                maxIndex = torch.argmax(pFit)
                choose = 0
                if (minIndex < 0.5 * pop and maxIndex < 0.5 * pop):
                    choose = 1
                elif (minIndex > 0.5 * pop and maxIndex < 0.5 * pop):
                    choose = 2
                elif (minIndex < 0.5 * pop and maxIndex > 0.5 * pop):
                    choose = 3
                elif (minIndex > 0.5 * pop and maxIndex > 0.5 * pop):
                    choose = 4
    ###############################################################################
                if choose < 3:
                    for i in range(int(pop / 2 + 1) - 1, pop):
                        x[i, :] = x[i, :] * (1 + torch.randn(1).cuda())
                        #                     print(x[i,:])
                        x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                    fit[int(pop / 2 + 1) -
                        1:pop] = self.predict_absorption_energy(
                            x[int(pop / 2 + 1) - 1:pop, :], index, M)
                    if choose == 1:
                        x[minIndex, :] = x[minIndex, :] * (
                            1 + torch.randn(1).cuda())
                        x[minIndex, :] = self.Bounds(x[minIndex, :], self.lb,
                                                     self.ub)
                        fit[minIndex] = self.predict_absorption_energy(
                            x[minIndex - 1:minIndex, :], index, M)
                    for i in range(0, int(0.5 * pop)):
                        if choose == 2 or minIndex != i:
                            # print(type(pop))
                            person = random.randint(0.5 * pop + 1, pop)
                            # print(person)
                            x[i, :] = x[i, :] + (pX[person, :] -
                                                 x[i, :]) * FL[i].cuda()
                            x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                    fit[0:int(0.5 * pop)] = self.predict_absorption_energy(
                        x[0:int(0.5 * pop), :], index, M)
                else:
                    for i in range(0, int(0.5 * pop)):
                        x[i, :] = x[i, :] * (1 + torch.randn(1).cuda())
                        x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                    fit[minIndex] = self.predict_absorption_energy(
                        x[minIndex - 1:minIndex, :], index, M)
                    if choose == 4:
                        x[minIndex, :] = x[minIndex, :] * (
                            1 + torch.randn(1).cuda())
                        x[minIndex, :] = self.Bounds(x[minIndex, :], self.lb,
                                                     self.ub)
                        fit[minIndex] = self.predict_absorption_energy(
                            x[minIndex - 1:minIndex, :], index, M)
                    for i in range(int(0.5 * pop), pop):
                        if choose == 3 or minIndex != i:
                            # print(type(pop))
                            person = random.randint(1, 0.5 * pop + 1)
                            # print(person)
                            x[i, :] = x[i, :] + (pX[person, :] -
                                                 x[i, :]) * FL[i].cuda()
                            x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                    fit[int(0.5 * pop):pop] = self.predict_absorption_energy(
                        x[int(0.5 * pop):pop, :], index, M)

    ###############################################################################
    #     Update the individual's best fitness value and the global best one
            for i in range(0, pop):
                if (fit[i] < pFit[i]):
                    pFit[i] = fit[i]
                    pX[i, :] = x[i, :]
                if (pFit[i] < fMin):
                    fMin = pFit[i]
            fMin = float(min(pFit))
            bestIndex = torch.argmin(pFit)
            bestX = pX[bestIndex, :]
            if index >= len(b2):
                b2 = torch.cat((b2, torch.zeros([M - len(b2), 1])), dim=0)
            b2[index] = fMin
            torch.save(x, "x10.pt")
            torch.save(fit, "fit10.pt")
            torch.save(pFit, "pFit10.pt")
            torch.save(pX, "pX10.pt")
            torch.save(fMin, "fMin10.pt")
            torch.save(bestIndex, "bestIndex10.pt")
            torch.save(bestX, "bestX10.pt")
            torch.save(b2, "b210.pt")
            torch.save(index, "index.pt")
        # print(fMin)
        return fMin, bestIndex, bestX, b2
