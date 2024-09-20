#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <vector>
using namespace std;

const int N = 10;
const int LIM = 10000;
using tiii = tuple<int, int, int>;

vector<string> G{
    "@.....#.#.",
    "#.#.#.#...",
    ".#.#..#.#.",
    ".....#..#.",
    "#.#.#..##.",
    "...#.#.#..",
    ".#.#.#..#.",
    "..#...#.#.",
    "#...#....#",
    "..#..#.#..",
};

vector<vector<pair<int, int>>> pos(80);
map<tiii, int> dp, op;
map<tiii, tiii> pr;
map<int, tiii> best;

void init() {
  map<pair<int, int>, int> dist;
  dist[{0, 0}] = 0;
  queue<pair<int, int>> q;
  q.push({0, 0});
  while (!q.empty()) {
    auto [r, c] = q.front();
    q.pop();
    for (auto [r2, c2] : {pair{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}) {
      if (0 <= r2 && r2 < N && 0 <= c2 && c2 < N && G[r2][c2] == '.' &&
          !dist.count({r2, c2})) {
        dist[{r2, c2}] = dist[{r, c}] + 1;
        pos[dist[{r2, c2}]].emplace_back(r2, c2);
        q.push({r2, c2});
      }
    }
  }
  vector<int> sz;
  for (auto l : pos) {
    sz.push_back(l.size());
  }
  vector<vector<tiii>> elts(62);
  dp[{0, 0, 1}] = 1;
  elts[1].emplace_back(0, 0, 1);
  for (int d = 0; d < 60; d++) {
    if (d > 1 && sz[d] == 0) {
      break;
    }
    while (!elts[d].empty()) {
      auto tt = elts[d].back();
      auto [u, v, state] = tt;
      elts[d].pop_back();
      if (0 <= u && u <= LIM && 0 <= v && v <= LIM) {
        if (!best.count(u)) {
          best[u] = tt;
        }
      } else {
        continue;
      }
      tiii t0 = tiii{u, v, state ^ 1}, t1, t2;
      if (!dp.count(t0)) {
        dp[t0] = d + 1;
        op[t0] = 0;
        pr[t0] = tt;
        elts[d + 1].push_back(t0);
      }
      if (sz[d] >= 1) {
        if (state % 2) {
          t1 = tiii{v, v, state ^ 1};
        } else {
          t1 = tiii{u, u + v, state ^ 1};
        }
        if (!dp.count(t1)) {
          dp[t1] = d + 1;
          op[t1] = 1;
          pr[t1] = tt;
          elts[d + 1].push_back(t1);
        }
      }
      if (sz[d] >= 2) {
        if (state % 2) {
          t2 = tiii{u - 1, v, state ^ 1};
        } else {
          t2 = tiii{u + 1, v, state ^ 1};
        }
        if (!dp.count(t2)) {
          dp[t2] = d + 1;
          op[t2] = 2;
          pr[t2] = tt;
          elts[d + 1].push_back(t2);
        }
      }
    }
  }
}

void solve() {
  int K;
  cin >> K;

  vector<int> rev;
  for (auto state = best[K]; state != tiii{0, 0, 1}; state = pr[state]) {
    rev.push_back(op[state]);
  }
  rev.push_back(0);
  reverse(rev.begin(), rev.end());

  vector<string> out = G;
  for (int i = 0; i < (int)rev.size(); i++) {
    for (int x = 0; x < rev[i]; x++) {
      auto [r, c] = pos[i][x];
      out[r][c] = '*';
    }
  }
  cout << N << " " << N << endl;
  for (int i = 0; i < N; i++) {
    cout << out[i] << endl;
  }
}

int main() {
  init();
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": ";
    solve();
  }
  return 0;
}
