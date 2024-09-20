#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

int N, leaves;
vector<vector<int>> ch;
vector<unordered_set<string>> S(N);

// Collapse nodes with 1 ch, u->v->(w1,w2,...) into u->(w1, w2, ...).
void dfs1(int u) {
  if (ch[u].empty()) {
    leaves++;
    return;
  }
  while (ch[u].size() == 1) {
    int v = ch[u][0];
    S[u].insert(S[v].begin(), S[v].end());
    ch[u] = ch[v];
  }
  for (int v : ch[u]) {
    dfs1(v);
  }
}

int dfs2(int u, const string &s) {
  if (ch[u].empty()) {
    return S[u].count(s);
  }
  int has = 0;
  for (int v : ch[u]) {
    int res = dfs2(v, s);
    if (res == -1) {
      return -1;
    }
    has += res;
  }
  if (has == (int)ch[u].size()) {
    return 1;
  }
  if (has + 1 == (int)ch[u].size()) {
    return S[u].count(s);
  }
  return -1;
}

int solve() {
  cin >> N;
  ch.assign(N, {});
  S.assign(N, {});
  for (int i = 1, p; i < N; i++) {
    cin >> p;
    ch[p - 1].push_back(i);
  }
  unordered_map<string, int> freq;
  for (int i = 0, m; i < N; i++) {
    cin >> m;
    string s;
    for (int j = 0; j < m; j++) {
      cin >> s;
      S[i].insert(s);
      freq[s]++;
    }
  }
  // Collapse any single child chains in the tree, and count leaves.
  leaves = 0;
  dfs1(0);
  // Try all words with freq[k] >= L (there can only be O(W/L).
  int ans = 0;
  for (const auto &[s, f] : freq) {
    if (f >= leaves && dfs2(0, s) == 1) {
      ans++;
    }
  }
  return ans;
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}