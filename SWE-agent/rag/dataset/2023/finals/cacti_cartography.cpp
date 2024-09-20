#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>
using namespace std;

typedef long long int64;

const int64 kInf = 12345678912345LL;
const int64 MAXK = 50, KLIM = (MAXK + 1) << 1, NLIM = 1000;

int N, M, K;
vector<int> weights;
vector<vector<int>> adj;
int64 f[2][KLIM][KLIM][NLIM];
int timer;
vector<int> tin, tout, par;
vector<int> cycle_ids, cycle_lens, cycle_rev, dist_to_par;
vector<vector<int>> edge_lnks;
vector<pair<int, int>> dp_edges;

void dfs(int x) {
  tin[x] = timer++;
  for (int y : adj[x]) {
    if (tin[y] != -1) {
      continue;
    }
    dfs(y);
  }
  tout[x] = timer++;
}

int cycle_counter;
vector<int> dfs_stack;

void dfs_cycles(int x, int p) {
  tin[x] = timer++;
  dfs_stack.push_back(x);
  for (int y : adj[x]) {
    if (y == p) {
      continue;
    }
    if (y == par[x]) {
      int cur_len = 1;
      int rev_cycle = -1;
      for (int z = (int)dfs_stack.size() - 1; dfs_stack[z] != par[x]; z--) {
        int edge_id = edge_lnks[dfs_stack[z - 1]][dfs_stack[z]];
        int rev_edge_id = edge_lnks[dfs_stack[z]][dfs_stack[z - 1]];
        if (rev_edge_id != -1 && cycle_ids[rev_edge_id] != -1) {
          rev_cycle = cycle_ids[rev_edge_id];
          cycle_rev[rev_cycle] = cycle_counter;
        }
        cycle_ids[edge_id] = cycle_counter;
        dist_to_par[edge_id] = cur_len++;
      }
      cycle_lens.push_back(cur_len);
      cycle_rev.push_back(rev_cycle);
      cycle_counter++;
    } else {
      if (tin[y] != -1) {
        continue;
      }
      if (edge_lnks[x][y] == -1) {
        edge_lnks[x][y] = dp_edges.size();
        dp_edges.emplace_back(x, y);
        cycle_ids.push_back(-1);
        dist_to_par.push_back(-1);
      }
      dfs_cycles(y, x);
    }
  }
  dfs_stack.pop_back();
}

int lazy_max_dist;

int64 lazy_dp(int edge_id, int dist_to_last, int par_dist, int promised) {
  if (dist_to_last > K + 1) {
    promised = 1;
  }
  if (f[promised][dist_to_last][par_dist][edge_id] != -1LL) {
    return f[promised][dist_to_last][par_dist][edge_id];
  }
  const int base_cycle_id = cycle_ids[edge_id];
  if (!dist_to_last && base_cycle_id != -1 &&
      cycle_lens[base_cycle_id] - dist_to_par[edge_id] < par_dist) {
    return f[promised][dist_to_last][par_dist][edge_id] = lazy_dp(
               edge_id, dist_to_last,
               cycle_lens[base_cycle_id] - dist_to_par[edge_id], promised);
  }
  if (!promised && base_cycle_id != -1 &&
      dist_to_par[edge_id] + par_dist < dist_to_last) {
    return f[promised][dist_to_last][par_dist][edge_id] = lazy_dp(
               edge_id, dist_to_par[edge_id] + par_dist, par_dist, promised);
  }
  const int x = dp_edges[edge_id].second;
  int64 res = kInf;
  if (dist_to_last) {
    res = min(res, lazy_dp(edge_id, 0, par_dist, 0) + weights[x]);
  }
  if (dist_to_last == lazy_max_dist) {
    return f[promised][dist_to_last][par_dist][edge_id] = res;
  }
  const int p = dp_edges[edge_id].first;
  vector<int> edge_list;
  vector<pair<int, int>> edge_groups;
  map<int, int> cycle_to_ind;
  for (int y : adj[x]) {
    if (y == p) {
      continue;
    }
    int new_edge_id = edge_lnks[x][y];
    if (new_edge_id == -1) {
      continue;
    }
    int cycle_id = cycle_ids[new_edge_id];
    if (cycle_id != -1) {
      auto mit = cycle_to_ind.find(cycle_id);
      if (mit != cycle_to_ind.end()) {
        edge_groups[mit->second].second = edge_list.size();
        edge_list.push_back(new_edge_id);
        continue;
      }
    }
    int rev_cycle_id = (cycle_id == -1 ? -1 : cycle_rev[cycle_id]);
    cycle_to_ind[rev_cycle_id] = edge_groups.size();
    edge_groups.emplace_back(edge_list.size(), -1);
    edge_list.push_back(new_edge_id);
  }
  const int esz = edge_list.size();
  if (!esz) {
    return f[promised][dist_to_last][par_dist][edge_id] =
               (promised || dist_to_last > K) ? res : 0LL;
  }
  const int egsz = edge_groups.size();
  vector<int64> edge_f(esz), edge_f2(esz);
  for (int new_dist_to_last = dist_to_last + 1, new_promised = promised;
       new_dist_to_last <= lazy_max_dist && res;
       ++new_dist_to_last, new_promised = 1) {
    const int inv_dist_to_last = lazy_max_dist - new_dist_to_last;
    const int min_dist_to_last = min(dist_to_last + 1, 2 + inv_dist_to_last);
    for (int i = 0; i < esz; ++i) {
      const int new_edge_id = edge_list[i];
      const int new_cycle_id = cycle_ids[new_edge_id];
      int new_par_dist = lazy_max_dist;
      if (new_cycle_id != -1) {
        if (new_cycle_id == base_cycle_id) {
          new_par_dist = par_dist;
          new_par_dist =
              min(new_par_dist, 1 + cycle_lens[base_cycle_id] -
                                    dist_to_par[edge_id] + inv_dist_to_last);
        } else {
          new_par_dist = min_dist_to_last - 1;
        }
      }
      edge_f[i] = lazy_dp(new_edge_id, min_dist_to_last, new_par_dist, 0);
    }
    for (int i = 0; i < esz; i++) {
      const int new_edge_id = edge_list[i];
      const int new_cycle_id = cycle_ids[new_edge_id];
      int new_par_dist = lazy_max_dist;
      if (new_cycle_id != -1) {
        if (new_cycle_id == base_cycle_id) {
          new_par_dist = par_dist;
        } else {
          new_par_dist = dist_to_last;
        }
      }
      edge_f2[i] =
          lazy_dp(new_edge_id, new_dist_to_last, new_par_dist, new_promised);
    }
    int64 f_sum = 0LL, min_diff = kInf;
    for (int i = 0; i < egsz; i++) {
      const int ind1 = edge_groups[i].first;
      int64 edge_group_f = edge_f[ind1], edge_group_f2 = edge_f2[ind1];
      const int ind2 = edge_groups[i].second;
      if (ind2 != -1) {
        edge_group_f = min(edge_group_f, edge_f[ind2]);
        edge_group_f2 = min(edge_group_f2, edge_f2[ind2]);
      }
      min_diff = min(min_diff, edge_group_f2 - edge_group_f);
      f_sum += edge_group_f;
    }
    res = min(res, f_sum + min_diff);
  }
  return f[promised][dist_to_last][par_dist][edge_id] = res;
}

int64 solve() {
  vector<pair<int, int>> edges;
  cin >> N >> M >> K;
  weights.resize(N);
  for (int i = 0; i < N; ++i) {
    cin >> weights[i];
  }
  edges.resize(M);
  for (int i = 0; i < M; ++i) {
    cin >> edges[i].first >> edges[i].second;
    --edges[i].first;
    --edges[i].second;
  }
  int root = N;
  edges.emplace_back(N++, 0);
  M++;
  adj.assign(N, {});
  for (int i = 0; i < M; i++) {
    adj[edges[i].first].push_back(edges[i].second);
    adj[edges[i].second].push_back(edges[i].first);
  }
  timer = 0;
  tin.assign(N, -1);
  tout.resize(N, -1);
  dfs(root);
  vector<int> tin0 = tin, tout0 = tout;
  for (int x = 0; x < N; x++) {
    reverse(adj[x].begin(), adj[x].end());
  }
  timer = 0;
  tin.assign(N, -1);
  tout.resize(N, -1);
  dfs(root);
  par.assign(N, -1);
  for (int x = 0; x < N; x++) {
    for (int y : adj[x]) {
      if (tin0[y] < tin0[x] && tin[y] < tin[x]) {
        par[x] = y;
      }
    }
  }
  cycle_counter = 0;
  dp_edges.clear();
  edge_lnks.assign(N, vector<int>(N, -1));
  cycle_ids.clear();
  cycle_lens.clear();
  cycle_rev.clear();
  dist_to_par.clear();
  timer = 0;
  tin.assign(N, -1);
  dfs_stack.clear();
  dfs_cycles(root, root);
  for (int x = 0; x < N; x++) {
    reverse(adj[x].begin(), adj[x].end());
  }
  timer = 0;
  tin.assign(N, -1);
  dfs_cycles(root, root);
  memset(f, -1, sizeof f);
  lazy_max_dist = ((K + 1) << 1) - 1;
  int64 res = kInf;
  for (int i = 0; i <= K; i++) {
    res = min(res, lazy_dp(edge_lnks[root][0], i + K + 1, lazy_max_dist, 1));
  }
  return res;
}

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}
