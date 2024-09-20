#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

const int N = 13;

vector<string> G;
vector<int> b;
vector<vector<int>> len;
vector<vector<pair<int, int>>> cells;

inline bool check(int r, int c) {
  return r >= 0 && c >= 0 && r < N && c < N;
}

void dfs(int r, int c, int l) {
  len[r][c] = l;
  if (static_cast<int>(cells.size()) <= l) {
    cells.resize(l + 1);
  }
  cells[l].emplace_back(r, c);
  for (auto [r2, c2] : {pair{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}) {
    if (!check(r2, c2) || G[r2][c2] == '#' || len[r2][c2] != -1) {
      continue;
    }
    dfs(r2, c2, l + 1);
  }
}

int count_adj(int r, int c) {
  int res = 0;
  for (auto [r2, c2] : {pair{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}) {
    if (!check(r2, c2)) {
      continue;
    }
    res += G[r2][c2] == '#';
  }
  return res;
}

void run(int r, int c, int l) {
  len[r][c] = l;
  if (l & 1) {
    for (auto [r2, c2] : {pair{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}) {
      if (!check(r2, c2) || G[r2][c2] != '#') {
        continue;
      }
      if (count_adj(r2, c2) == 3) {
        G[r2][c2] = '.';
        break;
      }
    }
  }
  for (auto [r2, c2] : {pair{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}) {
    if (!check(r2, c2) || G[r2][c2] == '#' || len[r2][c2] != -1) {
      continue;
    }
    run(r2, c2, l + 1);
  }
}

void solve() {
  int K;
  cin >> K;
  b.clear();
  for (int x = K; x; x >>= 1) {
    b.push_back(x & 1);
  }
  reverse(b.begin(), b.end());
  G.assign(N, string(N, '.'));
  G[0][0] = '@';
  for (int j = 1, par = 0; j < N; j += 3, par ^= 1) {
    int st = 0, fn = N;
    if (par) {
      st += 1;
    } else {
      fn -= 1;
    }
    for (int i = st; i < fn; i++) {
      G[i][j] = '#';
      G[i][j + 1] = '#';
    }
  }
  len.assign(N, vector<int>(N, -1));
  run(0, 0, 0);
  len.assign(N, vector<int>(N, -1));
  cells.clear();
  dfs(0, 0, 0);
  vector<int> d0;
  for (int i = 2; i < (int)cells.size(); i += 2) {
    if (d0.size() & 1) {
      d0.push_back(i);
      continue;
    }
    if (cells[i].size() > 1) {
      d0.push_back(i);
    }
  }
  for (int i = 0, ind = 0; i < (int)b.size(); i++, ind += 2) {
    if (b[i]) {
      int d = d0[ind];
      for (int j = 0; j <= 1; j++) {
        auto [r, c] = cells[d][j];
        G[r][c] = '*';
      }
    }
    for (int j = 0; j <= 1; j++) {
      int d = d0[ind + 1] + j;
      auto [r, c] = cells[d][0];
      G[r][c] = '*';
    }
  }
  cout << N << " " << N << endl;
  for (int i = 0; i < N; i++) {
    cout << G[i] << endl;
  }
}

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": ";
    solve();
  }
  return 0;
}
