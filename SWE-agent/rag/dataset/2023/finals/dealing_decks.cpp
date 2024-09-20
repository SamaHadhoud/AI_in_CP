#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

const int kBitCount = 22;

class PersistentTrie {
  vector<int> roots, last, last_rec;
  vector<vector<int>> nodes;

  inline int get_root(int ind) {
    return ind == -1 ? 0 : roots[ind];
  }

  vector<int> get_bits(int val) {
    vector<int> bits(kBitCount, 0);
    for (int i = 0; i < kBitCount; i++) {
      if (val & (1 << i)) {
        bits[kBitCount - i - 1] = 1;
      }
    }
    return bits;
  }

 public:
  void reset(int n) {
    last.assign(1, -1);
    last_rec.assign(1, -1);
    nodes.assign(2, vector<int>(1, -1));
    roots.clear();
    roots.reserve(n + 1);
    last.reserve((n + 1) * (kBitCount + 1));
    last_rec.reserve((n + 1) * (kBitCount + 1));
    nodes[0].reserve((n + 1) * (kBitCount + 1));
    nodes[1].reserve((n + 1) * (kBitCount + 1));
  }

  void add(int val, int ind) {
    vector<int> bits = get_bits(val);
    int r = roots.size(), node_id = nodes[0].size();
    roots.push_back(node_id);
    int root_id = get_root(r - 1);
    nodes[0].push_back(nodes[0][root_id]);
    nodes[1].push_back(nodes[1][root_id]);
    last.push_back(ind);
    vector<int> visited_nodes(1, node_id);
    visited_nodes.reserve(kBitCount + 1);
    for (int i = 0; i < kBitCount; ++i) {
      int next_node_id = nodes[bits[i]][node_id];
      if (next_node_id == -1) {
        nodes[0].push_back(-1);
        nodes[1].push_back(-1);
      } else {
        nodes[0].push_back(nodes[0][next_node_id]);
        nodes[1].push_back(nodes[1][next_node_id]);
      }
      last.push_back(ind);
      next_node_id = (int)nodes[0].size() - 1;
      nodes[bits[i]][node_id] = next_node_id;
      node_id = next_node_id;
      visited_nodes.push_back(node_id);
    }
    last_rec.resize(last.size());
    last_rec[visited_nodes.back()] = last[visited_nodes.back()];
    for (int i = (int)visited_nodes.size() - 2; i >= 0; i--) {
      const int node_id = visited_nodes[i];
      last_rec[node_id] = last[node_id];
      for (int j = 0; j < 2; ++j) {
        const int child_node_id = nodes[j][node_id];
        if (child_node_id == -1) {
          last_rec[node_id] = -1;
        } else {
          last_rec[node_id] = min(last_rec[node_id], last_rec[child_node_id]);
        }
      }
    }
  }

  int get_mex(int l, int r, int val) {
    if (!r) {
      return val ? 0 : 1;
    }
    int res = 0;
    vector<int> bits = get_bits(val);
    int node_id = get_root(r);
    for (int i = 0; i < kBitCount; ++i) {
      if (node_id == -1) {
        break;
      }
      for (int j = 0; j < 2; ++j) {
        const int bit_val = bits[i] ^ j;
        const int child_node_id = nodes[bit_val][node_id];
        if (child_node_id == -1) {
          node_id = child_node_id;
          res |= (j << (kBitCount - i - 1));
          break;
        }
        if (last_rec[child_node_id] < l) {
          node_id = child_node_id;
          res |= (j << (kBitCount - i - 1));
          break;
        }
      }
    }
    return res;
  }
};

PersistentTrie trie;

long long solve() {
  int N, x1, y1, z1, x2, y2, z2, x3, y3, z3;
  cin >> N >> x1 >> y1 >> z1 >> x2 >> y2 >> z2 >> x3 >> y3 >> z3;
  vector<int> A(N), B(N), C(N);
  long long pa = 0LL, pb = 0LL, pc = 0LL;
  for (int i = 0; i < N; i++) {
    pa = (pa * x1 + y1) % z1;
    pb = (pb * x2 + y2) % z2;
    pc = (pc * x3 + y3) % z3;
    A[i] = min(i + 1, (int)(1 + pa));
    B[i] = max(A[i], (int)(i + 1 - pb));
    C[i] = min(i, (int)pc);
  }
  A.insert(A.begin(), 0);
  B.insert(B.begin(), 0);
  C.insert(C.begin(), 0);
  trie.reset(N);
  trie.add(0, 0);
  vector<int> f(N + 1, 0);
  for (int i = 1; i <= N; ++i) {
    f[i] = trie.get_mex(i - B[i], i - A[i], f[C[i]]);
    trie.add(f[i], i);
  }
  vector<int> lnk(1 << kBitCount, -1);
  long long ans = 0LL;
  for (int i = 1; i <= N; i++) {
    if (lnk[f[i]] == -1) {
      lnk[f[i]] = i;
    }
    ans += lnk[f[i]];
  }
  return ans;
}

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}
