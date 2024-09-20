#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int N, M;
vector<vector<int>> G;

inline bool check(int r, int c) {
  return r >= 0 && c >= 0 && r < N && c < M;
}

inline void upd_options(int r, int c, vector<pair<int, int>>& options) {
  options.emplace_back(r, c);
  for (auto [r2, c2] : {pair{r - 2, c}, {r - 1, c}, {r, c - 1}, {r, c - 2}}) {
    if (r2 >= 0 && c2 >= 0) {
      options.emplace_back(r2, c2);
    }
  }
}

int count(vector<pair<int, int>>& options) {
  sort(options.begin(), options.end());
  int res = 0;
  for (int i = 0; i < (int)options.size(); i++) {
    if (i > 0 && options[i] == options[i - 1]) {
      continue;
    }
    auto [r, c] = options[i];
    res += (c + 2 >= M) ? 0 : G[r][c] == G[r][c + 1] && G[r][c] == G[r][c + 2];
    res += (r + 2 >= N) ? 0 : G[r][c] == G[r + 1][c] && G[r][c] == G[r + 2][c];
  }
  return res;
}

int solve() {
  cin >> N >> M;
  G.assign(N, vector<int>(M));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      cin >> G[i][j];
    }
  }
  vector<vector<int>> b(N, vector<int>(M));
  vector<int> cnts(1 << 7, 0);
  vector<pair<int, int>> options, semi_options;
  int res = 0;
  int skip_same = (N > 4 || M > 4);
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < M; c++) {
      semi_options.clear();
      upd_options(r, c, semi_options);
      for (auto [r2, c2] : {pair{r + 1, c}, {r, c + 1}}) {
        if (!check(r2, c2) || G[r][c] == G[r2][c2]) {
          continue;
        }
        swap(G[r][c], G[r2][c2]);
        options = semi_options;
        upd_options(r2, c2, options);
        b[r][c] = max(b[r][c], count(options));
        swap(G[r][c], G[r2][c2]);
      }
      if (skip_same) {
        res = max(res, b[r][c]);
      }
      cnts[b[r][c]]++;
    }
  }
  int max_cnt = 0;
  for (int i = 0; i < (int)cnts.size(); i++) {
    if (cnts[i]) {
      max_cnt = max(max_cnt, i);
    }
  }
  const int WINSZ = 3;
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < M; c++) {
      if (res == 16) {
        break;
      }
      if (b[r][c] + 8 <= res) {
        continue;
      }
      if (skip_same) {
        bool found = false;
        for (auto [r11, c11] : {pair{r + 1, c}, {r, c + 1}}) {
          if (!check(r11, c11)) {
            continue;
          }
          if (G[r][c] != G[r11][c11]) {
            found = true;
            break;
          }
        }
        if (!found) {
          continue;
        }
      }
      for (int r2 = max(0, r - WINSZ); r2 <= min(r + WINSZ, N - 1); r2++) {
        for (int c2 = max(0, c - WINSZ); c2 <= min(c + WINSZ, M - 1); c2++) {
          cnts[b[r2][c2]]--;
        }
      }
      int cur_max = 0;
      for (int i = max_cnt; i > 0; i--) {
        if (cnts[i]) {
          cur_max = i;
          break;
        }
      }
      res = max(res, cur_max + b[r][c]);
      for (auto [r11, c11] : {pair{r + 1, c}, {r, c + 1}}) {
        if (!check(r11, c11) || (skip_same && G[r][c] == G[r11][c11])) {
          continue;
        }
        swap(G[r][c], G[r11][c11]);
        semi_options.clear();
        upd_options(r, c, semi_options);
        upd_options(r11, c11, semi_options);
        for (int r2 = max(0, r - WINSZ); r2 <= min(r + WINSZ, N - 1); r2++) {
          for (int c2 = max(0, c - WINSZ); c2 <= min(c + WINSZ, M - 1); c2++) {
            for (auto [r22, c22] : {pair{r2 + 1, c2}, {r2, c2 + 1}}) {
              if (!check(r22, c22) || (skip_same && G[r2][c2] == G[r22][c22])) {
                continue;
              }
              swap(G[r2][c2], G[r22][c22]);
              options = semi_options;
              upd_options(r2, c2, options);
              upd_options(r22, c22, options);
              res = max(res, count(options));
              swap(G[r2][c2], G[r22][c22]);
            }
          }
        }
        swap(G[r][c], G[r11][c11]);
      }
      for (int r2 = max(0, r - WINSZ); r2 <= min(r + WINSZ, N - 1); r2++) {
        for (int c2 = max(0, c - WINSZ); c2 <= min(c + WINSZ, M - 1); c2++) {
          cnts[b[r2][c2]]++;
        }
      }
    }
  }
  return res;
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}
