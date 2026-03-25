from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def get_bond_id(mol, i, j, bdlall):
    if isinstance(bdlall, dict):
        ind = bdlall.get((mol, i, j))
        if ind is None:
            ind = bdlall.get((mol, j, i))
        if ind is not None:
            return ind
        print('-  an error case for {!r}.........'.format([mol, i, j]))
        return None

    bdl = [mol, i, j]
    bdlr = [mol, j, i]
    if bdl in bdlall:
        ind = bdlall.index(bdl)
    elif bdlr in bdlall:
        ind = bdlall.index(bdlr)
    else:
        print('-  an error case for {!r}.........'.format(bdl))
        print(bdlall)
        ind = None
    return ind


class links(object):
  ''' links the bonds angles and torsions into a big graph '''
  def __init__(self, species=None, bonds=None, angs=None,
               tors=None, hbs=None,  # g=False,
               vdwcut=None,
               molecules=None,
               progress=True):
      self.dft_energy = {}
      self.species = [] if species is None else list(species)
      self.bonds = [] if bonds is None else list(bonds)
      self.angs = [] if angs is None else list(angs)
      self.tors = [] if tors is None else list(tors)
      self.hbs = [] if hbs is None else list(hbs)
      self.vdwcut = vdwcut
      self.progress = progress

      self._bond_types = set(self.bonds)
      self._angle_types = set(self.angs)
      self._torsion_types = set(self.tors)
      self._hb_types = set(self.hbs)
      molecule_names = [] if molecules is None else list(molecules.keys())
      self._molecule_names = molecule_names

      self.update_links({} if molecules is None else molecules)

      if molecules is not None:
          for mol in molecule_names:
              self.dft_energy[mol] = molecules[mol].energy_nw
      # self.histogram()

  @staticmethod
  def _format_seconds(seconds):
      seconds = max(0.0, float(seconds))
      if seconds < 60.0:
          return '{:.1f}s'.format(seconds)
      minutes, seconds = divmod(seconds, 60.0)
      if minutes < 60.0:
          return '{:d}m {:.1f}s'.format(int(minutes), seconds)
      hours, minutes = divmod(minutes, 60.0)
      return '{:d}h {:d}m {:.1f}s'.format(int(hours), int(minutes), seconds)

  def _stage_start(self, name, total):
      stage = {
          'name': name,
          'total': total,
          'start': time.time(),
          'last_report': 0,
          'report_every': max(1, total // 10) if total else 1,
      }
      if self.progress:
          print('-  [links] {:s}: starting ({:d} molecules) ...'.format(name, total))
      return stage

  def _stage_tick(self, stage, done):
      if not self.progress:
          return
      total = stage['total']
      if total == 0 or done == 0:
          return
      if done != total and done - stage['last_report'] < stage['report_every']:
          return
      elapsed = time.time() - stage['start']
      eta = 0.0 if done == 0 else elapsed * max(total - done, 0) / float(done)
      print('-  [links] {name:s}: {done:d}/{total:d} molecules, elapsed {elapsed:s}, eta {eta:s}'.format(
          name=stage['name'],
          done=done,
          total=total,
          elapsed=self._format_seconds(elapsed),
          eta=self._format_seconds(eta),
      ))
      stage['last_report'] = done

  def _stage_finish(self, stage, extra=''):
      if not self.progress:
          return
      elapsed = time.time() - stage['start']
      message = '-  [links] {:s}: done in {:s}'.format(stage['name'], self._format_seconds(elapsed))
      if extra:
          message += ' ({:s})'.format(extra)
      print(message)

  @staticmethod
  def _empty_index_array():
      return np.zeros((0, 1), dtype=np.int64)

  def _bond_type(self, atom_i, atom_j):
      forward = atom_i + '-' + atom_j
      reverse = atom_j + '-' + atom_i
      if forward in self._bond_types:
          return forward
      if reverse in self._bond_types:
          return reverse
      raise RuntimeError('-  Error: bond {:s} not found in bond list.'.format(forward))

  def _require_bond_index(self, mol, iatom, jatom):
      ind = self._bond_index.get((mol, iatom, jatom))
      if ind is None:
          raise RuntimeError('-  Error: bond link for {:s} ({:d}, {:d}) not found.'.format(mol, iatom, jatom))
      return ind

  def update_links(self, molecules):
      total_start = time.time()
      self.get_bond_link(molecules)
      self.get_atom_link(molecules)
      self.get_angle_link(molecules)
      self.get_torsion_link(molecules)
      self.get_vdw_link(molecules)
      self.get_hb_link(molecules)
      if hasattr(self, '_atom_index'):
          del self._atom_index
      if hasattr(self, '_bond_index'):
          del self._bond_index
      if self.progress:
          print('-  [links] total: done in {:s}'.format(self._format_seconds(time.time() - total_start)))
      # if self.g:
      #    self.get_glink()

  def get_atom_link(self, molecules):
      ''' atoms are labeled by different molecules '''
      stage = self._stage_start('atom', len(molecules))
      self.atomlist = {sp: [] for sp in self.species}
      self.atlab = {sp: [] for sp in self.species}
      self.nsp = {}
      extra_species = []

      for mol_index, mol in enumerate(list(molecules.keys()), start=1):
          atom_names = molecules[mol].atom_name
          for iatom, sp in enumerate(atom_names):
              if sp not in self.atomlist:
                  self.atomlist[sp] = []
                  self.atlab[sp] = []
                  extra_species.append(sp)
              global_index = self._atom_index[(mol, iatom)]
              label = [mol, iatom]
              self.atomlist[sp].append([global_index])
              self.atlab[sp].append(label)
          self._stage_tick(stage, mol_index)

      species_order = list(self.species) + [sp for sp in extra_species if sp not in self.species]
      self.atlall = []                                # bond label concated all together
      atlall_index = {}
      for sp in species_order:
          for label in self.atlab.get(sp, []):
              atlall_index[(label[0], label[1])] = len(self.atlall)
              self.atlall.append(label)
          self.nsp[sp] = len(self.atlab.get(sp, []))

      self.dalink = []                                # link the atom divided by specices to a molecule
      for mol, iatom in self.atom_lab:
          self.dalink.append([atlall_index[(mol, iatom)]])

      self.atomlink = {mol: [] for mol in molecules}
      for i, atl in enumerate(self.atlall):
          self.atomlink[atl[0]].append([i])

      self._stage_finish(stage, extra='{:d} species groups, {:d} atoms'.format(len(species_order), len(self.atlall)))

#   def get_glink(self):
#       self.natom = len(self.atlall)
#       glist = [[] for a in self.atlall]
#       maxg  = 0
#
#       for ia,ang in enumerate(self.angall): # [key,ai,aj,ak]
#           atl = [ang[0],ang[2]]
#           i_  = self.atlall.index(atl)
#           glist[i_].append(ia+1)
#
#       for i,a in enumerate(self.atlall):
#           ng = len(glist[i])
#           if ng>maxg: maxg=ng
#
#       self.maxg    = maxg
#       self.glist   = np.zeros([self.natom,self.maxg,1],dtype=np.int64)
#       self.rijlist = np.zeros([self.natom,self.maxg,1],dtype=np.int64)
#       self.riklist = np.zeros([self.natom,self.maxg,1],dtype=np.int64)
#
#       for atom,gl in enumerate(glist):
#           for ng,g in enumerate(gl):
#               self.glist[atom][ng][0] = g  # gather theta and R is enough
#               # print('-  info:',maxg,g,self.atlall[atom],self.angall[g-1])
#               key,atomj,atomi,atomk = self.angall[g-1]
#
#               ij = get_bond_id(key,atomi,atomj,self.bdlall)
#               ik = get_bond_id(key,atomi,atomk,self.bdlall)
#
#               self.rijlist[atom][ng][0] = ij
#               self.riklist[atom][ng][0] = ik

  def get_bond_link(self, molecules):
      ''' label bonds and atoms '''
      stage = self._stage_start('bond', len(molecules))
      self.max_nei = 0
      self.nbond = 0
      self.nbd = {}
      self.bdlab = {bd: [] for bd in self.bonds}
      self.atom_lab = []
      self.rbd = {bd: [] for bd in self.bonds}
      self.natom = 0

      atom_index = {}
      dilink_buffer = {bd: [] for bd in self.bonds}
      djlink_buffer = {bd: [] for bd in self.bonds}

      for mol_index, key in enumerate(list(molecules.keys()), start=1):
          data = molecules[key]
          self.natom += data.natom
          self.nbond += data.nbond
          if data.max_nei > self.max_nei:    # get max neighbor
              self.max_nei = data.max_nei

          for iatom in range(data.natom):
              atom_index[(key, iatom)] = len(self.atom_lab)
              self.atom_lab.append([key, iatom])

          for i, bond in enumerate(data.bond):
              b0 = int(bond[0])
              b1 = int(bond[1])
              iatom = int(np.mod(b0, data.natom))
              jatom = int(np.mod(b1, data.natom))
              bd = self._bond_type(data.atom_name[iatom], data.atom_name[jatom])
              self.rbd[bd].append(data.rbd[:, i])
              self.bdlab[bd].append([key, b0, b1])
              dilink_buffer[bd].append([atom_index[(key, iatom)]])
              djlink_buffer[bd].append([atom_index[(key, jatom)]])

          self._stage_tick(stage, mol_index)

      self._atom_index = atom_index
      self.bdlall = []      # bond label concated all together
      self.bdlink = {mol: [] for mol in molecules}
      self.dilink, self.djlink = {}, {}
      bond_index = {}
      offset = 0

      for bd in self.bonds:
          self.nbd[bd] = len(self.bdlab[bd])
          if dilink_buffer[bd]:
              self.dilink[bd] = np.array(dilink_buffer[bd], dtype=np.int64)
              self.djlink[bd] = np.array(djlink_buffer[bd], dtype=np.int64)
          else:
              self.dilink[bd] = self._empty_index_array()
              self.djlink[bd] = self._empty_index_array()

          for local_index, bond_label in enumerate(self.bdlab[bd]):
              global_index = offset + local_index
              self.bdlall.append(bond_label)
              mol = bond_label[0]
              natom = molecules[mol].natom
              iatom = int(np.mod(bond_label[1], natom))
              jatom = int(np.mod(bond_label[2], natom))
              bond_index[(mol, iatom, jatom)] = global_index
              bond_index[(mol, jatom, iatom)] = global_index
              self.bdlink[mol].append([global_index])
          offset += self.nbd[bd]

      self._bond_index = bond_index
      self.dlist = np.zeros([self.natom, self.max_nei, 1], dtype=np.int64)
      self.dalist = np.zeros([self.natom, self.max_nei, 1], dtype=np.int64)

      for i, label in enumerate(self.atom_lab):
          mol = label[0]
          iatom = label[1]
          natom = molecules[mol].natom
          for j, jatom in enumerate(molecules[mol].table[iatom]):
              jatom_ = int(np.mod(jatom, natom))
              ind = self._require_bond_index(mol, iatom, jatom_)
              self.dlist[i][j][0] = ind + 1                              # bonds table for atom i
              self.dalist[i][j][0] = self._atom_index[(mol, jatom_)]     # atoms that bonded to atom i

      self._stage_finish(stage, extra='{:d} atoms, {:d} grouped bonds'.format(self.natom, len(self.bdlall)))

  def get_angle_link(self, molecules):
      stage = self._stage_start('angle', len(molecules))
      self.nang = {}
      self.anglab = {ang: [] for ang in self.angs}
      self.dglist = {ang: [] for ang in self.angs}
      self.dgilist = {ang: [] for ang in self.angs}
      self.dgklist = {ang: [] for ang in self.angs}
      self.boaij = {ang: [] for ang in self.angs}
      self.boajk = {ang: [] for ang in self.angs}
      self.theta = {ang: [] for ang in self.angs}
      self.angall = []
      self.anglink = {mol: {} for mol in molecules}

      for mol_index, key in enumerate(list(molecules.keys()), start=1):
          data = molecules[key]
          natom = data.natom
          atom_names = data.atom_name
          for i in range(len(data.ang_i)):
              ai = int(data.ang_i[i])
              aj = int(data.ang_j[i])
              ak = int(data.ang_k[i])
              ai_ = int(np.mod(ai, natom))
              aj_ = int(np.mod(aj, natom))
              ak_ = int(np.mod(ak, natom))

              an = atom_names[ai_] + '-' + atom_names[aj_] + '-' + atom_names[ak_]
              anr = atom_names[ak_] + '-' + atom_names[aj_] + '-' + atom_names[ai_]
              if an in self._angle_types:
                  ang = an
              elif anr in self._angle_types:
                  ang = anr
                  ai, aj, ak = int(data.ang_k[i]), int(data.ang_j[i]), int(data.ang_i[i])
                  ai_ = int(np.mod(ai, natom))
                  aj_ = int(np.mod(aj, natom))
                  ak_ = int(np.mod(ak, natom))
              else:
                  continue

              link_index = len(self.anglab[ang])
              self.anglab[ang].append([key, ai, aj, ak])
              self.boaij[ang].append([self._require_bond_index(key, ai_, aj_) + 1])
              self.boajk[ang].append([self._require_bond_index(key, aj_, ak_) + 1])
              self.dglist[ang].append([self._atom_index[(key, aj_)]])
              self.dgilist[ang].append([self._atom_index[(key, ai_)]])
              self.dgklist[ang].append([self._atom_index[(key, ak_)]])
              self.theta[ang].append(data.theta[i])
              self.anglink[key].setdefault(ang, []).append([link_index])
          self._stage_tick(stage, mol_index)

      for ang in self.angs:
          self.angall.extend(self.anglab[ang])
          self.nang[ang] = len(self.anglab[ang])

      for mol in molecules:
          for ang in self.angs:
              if self.nang[ang] > 0 and ang not in self.anglink[mol]:
                  self.anglink[mol][ang] = []

      total_angles = sum(self.nang.values()) if self.nang else 0
      self._stage_finish(stage, extra='{:d} grouped angles'.format(total_angles))

  def get_torsion_link(self, molecules):
      stage = self._stage_start('torsion', len(molecules))
      self.ntor = {}
      self.torlab = {tor: [] for tor in self.tors}
      self.tij = {tor: [] for tor in self.tors}
      self.tjk = {tor: [] for tor in self.tors}
      self.tkl = {tor: [] for tor in self.tors}
      self.dtj = {tor: [] for tor in self.tors}
      self.dtk = {tor: [] for tor in self.tors}
      self.s_ijk = {tor: [] for tor in self.tors}
      self.s_jkl = {tor: [] for tor in self.tors}
      self.w = {tor: [] for tor in self.tors}
      self.torlink = {mol: {} for mol in molecules}

      for mol_index, key in enumerate(list(molecules.keys()), start=1):
          data = molecules[key]
          natom = data.natom
          atom_names = data.atom_name
          for i in range(len(data.tor_i)):
              ti = int(data.tor_i[i])
              tj = int(data.tor_j[i])
              tk = int(data.tor_k[i])
              tl = int(data.tor_l[i])
              ti_ = int(np.mod(ti, natom))
              tj_ = int(np.mod(tj, natom))
              tk_ = int(np.mod(tk, natom))
              tl_ = int(np.mod(tl, natom))

              tn = atom_names[ti_] + '-' + atom_names[tj_] + '-' + atom_names[tk_] + '-' + atom_names[tl_]
              tnr = atom_names[tl_] + '-' + atom_names[tk_] + '-' + atom_names[tj_] + '-' + atom_names[ti_]
              if tn in self._torsion_types:
                  tor = tn
              elif tnr in self._torsion_types and tn != tnr:
                  tor = tnr
                  ti, tj = int(data.tor_l[i]), int(data.tor_k[i])
                  tk, tl = int(data.tor_j[i]), int(data.tor_i[i])
                  ti_ = int(np.mod(ti, natom))
                  tj_ = int(np.mod(tj, natom))
                  tk_ = int(np.mod(tk, natom))
                  tl_ = int(np.mod(tl, natom))
              else:
                  continue

              link_index = len(self.torlab[tor])
              self.torlab[tor].append([key, ti, tj, tk, tl])
              self.tij[tor].append([self._require_bond_index(key, ti_, tj_) + 1])
              self.tjk[tor].append([self._require_bond_index(key, tj_, tk_) + 1])
              self.tkl[tor].append([self._require_bond_index(key, tk_, tl_) + 1])
              self.dtj[tor].append([self._atom_index[(key, tj_)]])
              self.dtk[tor].append([self._atom_index[(key, tk_)]])
              self.s_ijk[tor].append(data.s_ijk[i])
              self.s_jkl[tor].append(data.s_jkl[i])
              self.w[tor].append(data.w[i])
              self.torlink[key].setdefault(tor, []).append([link_index])
          self._stage_tick(stage, mol_index)

      for tor in self.tors:
          self.ntor[tor] = len(self.torlab[tor])

      for mol in molecules:
          for tor in self.tors:
              if self.ntor[tor] > 0 and tor not in self.torlink[mol]:
                  self.torlink[mol][tor] = []

      total_torsions = sum(self.ntor.values()) if self.ntor else 0
      self._stage_finish(stage, extra='{:d} grouped torsions'.format(total_torsions))

  def get_vdw_link(self, molecules):
      stage = self._stage_start('vdw', len(molecules))
      self.vlab = {vb: [] for vb in self.bonds}
      self.nvb = {}
      self.rv = {vb: [] for vb in self.bonds}
      self.vi = {vb: [] for vb in self.bonds}
      self.vj = {vb: [] for vb in self.bonds}
      self.qij = {vb: [] for vb in self.bonds}
      self.vlink = {mol: {} for mol in molecules}

      for mol_index, key in enumerate(list(molecules.keys()), start=1):
          data = molecules[key]
          atom_names = data.atom_name
          for i, vi in enumerate(data.vi):
              vi = int(vi)
              vj = int(data.vj[i])
              vn = atom_names[vi] + '-' + atom_names[vj]
              vnr = atom_names[vj] + '-' + atom_names[vi]
              if vn in self._bond_types:
                  vb = vn
              elif vnr in self._bond_types:
                  vb = vnr
              else:
                  continue
              link_index = len(self.vlab[vb])
              self.rv[vb].append(data.rv[i, :])
              self.vlab[vb].append([key, vi, vj])
              self.qij[vb].append(data.q[:, vi] * data.q[:, vj] * 14.39975840)
              self.vi[vb].append([self._atom_index[(key, vi)]])
              self.vj[vb].append([self._atom_index[(key, vj)]])
              self.vlink[key].setdefault(vb, []).append([link_index])
          self._stage_tick(stage, mol_index)

      for vb in self.bonds:
          self.nvb[vb] = len(self.vlab[vb])

      for mol in molecules:
          for vb in self.bonds:
              if self.nvb[vb] > 0 and vb not in self.vlink[mol]:
                  self.vlink[mol][vb] = []

      total_vdw = sum(self.nvb.values()) if self.nvb else 0
      self._stage_finish(stage, extra='{:d} grouped vdw pairs'.format(total_vdw))

  def get_hb_link(self, molecules):
      stage = self._stage_start('hb', len(molecules))
      self.nhb = {}
      self.rhb = {}
      # self.rik     = {}
      # self.rij     = {}
      self.hblab = {}
      self.hij = {}
      self.hbthe, self.frhb = {}, {}
      self.hblink = {mol: {} for mol in molecules}

      for hb in self.hbs:
          self.hblab[hb] = []
          self.hbthe[hb] = []
          self.frhb[hb] = []
          self.rhb[hb] = []
          # self.rij[hb]   = []
          # self.rik[hb]   = []
          self.hij[hb] = []

      for mol_index, key in enumerate(list(molecules.keys()), start=1):
          data = molecules[key]
          atom_names = data.atom_name
          for i in range(len(data.hb_i)):
              hi = int(data.hb_i[i])
              hj = int(data.hb_j[i])
              hk = int(data.hb_k[i])
              hn = atom_names[hi] + '-' + atom_names[hj] + '-' + atom_names[hk]
              if hn not in self._hb_types:
                  continue
              link_index = len(self.hblab[hn])
              self.hij[hn].append([self._require_bond_index(key, hi, hj) + 1])
              self.hblab[hn].append([key, hi, hj, hk])
              self.rhb[hn].append(data.rhb[i, :])
              # self.rij[hb].append(molecules[key].rij[i,:])
              # self.rik[hb].append(molecules[key].rik[i,:])
              self.hbthe[hn].append(data.hbthe[i, :])
              self.frhb[hn].append(data.frhb[i, :])
              self.hblink[key].setdefault(hn, []).append([link_index])
          self._stage_tick(stage, mol_index)

      for hb in self.hbs:
          self.nhb[hb] = len(self.hblab[hb])
          self.rhb[hb] = np.array(self.rhb[hb])
          # self.rik[hb]   = np.array(self.rik[hb])
          self.frhb[hb] = np.array(self.frhb[hb])
          self.hbthe[hb] = np.array(self.hbthe[hb])

      for mol in molecules:
          for hb in self.hbs:
              if self.nhb[hb] > 0 and hb not in self.hblink[mol]:
                  self.hblink[mol][hb] = []

      total_hb = sum(self.nhb.values()) if self.nhb else 0
      self._stage_finish(stage, extra='{:d} grouped hb triples'.format(total_hb))

  def histogram(self):
      print('-  plotting bond length histogram ...')
      for bd in self.bonds:
          if self.nbd[bd] > 0:
              plt.figure()
              ax = plt.gca()
              ax.xaxis.set_major_locator(MultipleLocator(0.5))
              ax.xaxis.set_minor_locator(MultipleLocator(0.1))
              plt.ylabel('Bond length distribution')
              plt.xlabel('Radius (Angstrom)')
              plt.hist(np.reshape(self.rbd[bd], [-1]), 1000, alpha=0.01, label='%s' % bd)
              plt.legend(loc='best')
              plt.savefig('%s_bh.eps' % bd)
              plt.close()