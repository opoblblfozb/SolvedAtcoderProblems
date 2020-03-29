'''
title: format
unkown:条件を満たす(最小値、最大値)
algorithm:
otheridea:
'''


'''
title: Kleene Inversion
keyword: 転倒数, bisect_left
study:　注意すべき点、学んだことなどを言語化する。
・大きな方針までは、ぼやっとあっていた。
・このぼやっとした感じをより明確にすることで解答にたどり付けた。
・問題をMECEに分割できていれば。。。
・転倒数の求め方は、二重ループ回せば簡単
・実践では、転倒数の求め方に、妙にこだわってしまって、
・大局的な方針を洗練させるのに、時間をかけなかったことが問題
・bisectを使うことで高速化できそう。
'''
# B _再提出
n, k = map(int, input().split())
A = list(map(int, input().split()))
mod = 10 ** 9 + 7

naibu = 0
for i in range(n):
    for j in range(i+1, n):
        if A[i] > A[j]:
            naibu += 1
gaibu = 0
A2 = A
for i in range(n):
    for j in range(n):
        if A[i] > A2[j]:
            gaibu += 1

total = naibu*k + gaibu*(k*(k-1)//2)
print(total%mod)

# B _ 参考＿よくわからない。。；。
from bisect import bisect_left

N, K = map(int, input().split())
A = list(map(int, input().split()))
mod = 10**9 + 7

num = []
num2 = []
ret = 0
#### numをつくることでひとつのAでの転倒数。
for i in range(N):
    ind = N-1-i
    j = bisect_left(num, A[ind])
    num.insert(j, A[ind])
    ret += j

### 辞書を作っている、Aの出現頻度辞書
count = 0
a = {}
for i in range(N):
    if A[i] in a:
        a[A[i]] += 1
    else:
        a[A[i]] = 1

ret2 = N * (N-1)//2 - ret
for e in a.values():
    ret2 -= (e * (e-1))//2

ret *= K * (K+1)//2
ret2 *= K * (K-1)//2
ret += ret2
ret %= mod
print(ret)

'''
title: One-Stroke Path
keyword: 無向グラフ、　一筆書き、　permutations、　順列
study:　注意すべき点、学んだことなどを言語化する。
・データ構造⇒隣接行列　(無向グラフ他には、link型[(0とつながる)(１)(２)()])
・演算子*=
・データ構造と問題の解法には密接な関連がある
・問題をうまくとける形にデータを表しておくことが必要
'''
import itertools
n, m = map(int, input().split())
adj_matrix = [[0] * n for _ in range(n)]

for _ in range(m):
    a, b = map(int, input().split())
    adj_matrix[a-1][b-1] = 1
    adj_matrix[b-1][a-1] = 1

cnt = 0
#### permutationsで0から出発するありとあらゆる一筆つながりをつくって
#### adj_matrixを参照して、それが存在するかどうかを確認する。
#### 条件で絞りこむのではなく、条件を満たすものをつくって、それが存在するかどうか確認
for each in range(itertools.permutations(range(n))):
    if each[0] != 0:
        break
    flag = 1
    for i in range(n-1):
        flag *= adj_matrix[each[i]][each[i+1]]
    cnt += flag
print(cnt)

'''
title:硬貨の問題
keyword:貪欲法
study:
・貪欲法は、あるルールに基づいて、その場の最善を追求していく方法
・この問題自体は、硬貨の計算方法に注目
'''
Cs = list(map(int, input().split()))
A = int(input())
vals = [1, 5, 10, 50, 100, 500]

ans = 0
for i in reversed(range(len(vals))):
    t = min(A // vals[i], Cs[i])
    A = A - t*vals[i]
    ans += t
print(ans)

'''
title:A-おつり
keyword:貪欲法
study:
・上述の問題、硬貨の問題と本質的に同じ
'''
buy = int(input())
oturi = 1000-buy
vals = [1,5,10,50,100,500]

total = 0
for i in reversed(range(len(vals))):
    t = oturi // vals[i]
    oturi -= t*vals[i]
    total += t
print(total)

'''
title: 区間の問題
keyword: 貪欲法、区間スケジューリング問題
study:　注意すべき点、学んだことなどを言語化する。
区間の終端または始端でソートするは極めてよくみるテクニックらしい。

'''

n = int(input())
s = list(map(int, input().split()))
t = list(map(int, input().split()))

pair = []
for start, stop in zip(s, t):
    pair.append((start, stop))
pair = sorted(pair, key=lambda x:x[1])

ans = 0
now = 0
for work in pair:
    if now < work[0]:
        ans += 1
        now = work[1]
print(ans)


'''
title: A－東京都
keyword: 貪欲法、区間スケジューリング問題
study:　注意すべき点、学んだことなどを言語化する。
どうやったら、区間スケジューリング問題としてとらえられるのだろう。。。
わからない。。。
https://atcoder.jp/contests/kupc2015/tasks/kupc2015_a
'''
t = int(input())
for i in range(t):
    string = input()

    point = 0
    ans = 0
    while point < (len(string) - 4):
        if string[point:point+5] in ['tokyo', 'kyoto']:
            ans += 1
            point = point+5
        else:
            point += 1
    print(ans)

'''
title: Best Cow Linw
keyword: 貪欲法、辞書順最小の問題
study:　注意すべき点、学んだことなどを言語化する。
この問題は貪欲法っぽいね。
'''
n = int(input())
s = input()
t = ''

for _ in range(n):
    res = s[::-1]
    extract = sorted([s, res])[0]
    if extract == s:
        t += s[0]
        s = s[1:]
    elif extract == res:
        t += s[-1]
        s = s[:-1]

print(t)

'''
title: Saruman's Army
keyword: 貪欲法、条件の厳しい方を取っていく。
study:　注意すべき点、学んだことなどを言語化する。
処理的に難しい方針は、もっとすっきりできるor不正解である確率が高そう。。
http://poj.org/problem?id=3069
'''
n = int(input())
r = int(input())
x = list(map(int, input().split()))
x = sorted(x)

i = 0
ans = 0
while i < n:
    s = x[i]
    i += 1

    while i < n and x[i] <= s+r:
        i += 1

    p = x[i - 1]
    while i < n and x[i] <= p + r:
        i += 1

    ans += 1

print(ans)

'''
title: Multiple Gift
keyword: 貪欲法、条件の厳しい方を取っていく。
study:　注意すべき点、学んだことなどを言語化する。
一見条件が複雑だけれど、具体的な場合を考えて、法則を見つければ、
簡単に記述できる。
https://atcoder.jp/contests/abc083/tasks/arc088_a
'''
x, y = map(int, input().split())

a = []
t = x
while True:
    if t > y:
        break
    a.append(t)
    t *= 2

print(len(a))

'''
title: Fance Repair
keyword: 貪欲法、ハフマン符号
study:
あまり、教科書的な方法ではないけれど、
たぶん、教科書よりも早い解法で解けた。
プロコンの問題は、大学受験の数学問題よりも、より柔軟に解けそうだ。
'''
n = int(input())
ls = list(map(int, input().split()))
ls = sorted(ls, reverse=True)
nokori = sum(ls)

if n == 1:
    print(0)
    exit()

cost = 0
for i in range(n-1):
    cost += nokori
    nokori -= ls[i]
print(cost)

'''
title:　個数制限なしナップザック問題
keyword:　動的計画法
study:
動的計画法とは、つまるところ表にもとづく漸化式をいかに作るかという問題である。
'''
n = int(input())
wlist = [0]*n
vlist = [0]*n
for i in range(n):
    wlist[i], vlist[i] = map(int, input().split())
w = int(input())

for i in range(n):
    for j in range(w+1):
        if j < w[i]:
            dp[i+1][j] = dp[i][j]
        else:
            dp[i+1][j] = max(dp[i][j], dp[i+1][j-w[i]] + v[i])

########################################################################################################################################################################
############# ABC060
########################################################################################################################################################################
'''
title: Shiritori
keyword:
study:　注意すべき点、学んだことなどを言語化する。

'''
strings = list(input().split())

flag = 'YES'
for i in range(2):
    if strings[i][-1] != strings[i+1][0]:
        flag = 'NO'
print(flag)

'''
title: Choose Integers
keyword:
study:
割られる方の数が必ずAの倍数になるという条件の言い換え
mod的な性質

'''
A, B, C = list(map(int, input().split()))
flag = 'NO'
for i in range(1, B+1):
    if A*i % B == C:
        flag = 'YES'
print(flag)

'''
title: Choose Integers
keyword:
study:
割られる方の数が必ずAの倍数になるという条件の言い換え
mod的な性質

'''
n, t = map(int, input().split())
tlist = list(map(int, input().split()))

total = 0
for i in range(n):
    if i == n-1:
        total += T
    else:
        aida = tlist[i+1] - tlist[i]
        if aida <= T:
            total += aida
        else:
            total += T

'''
title: Simple Knapsack
keyword: ナップザック問題
study:
普通に動的計画法でだしたら、TLE食らった。
トンカチをもったらすべてくぎに見えるというのは避けたい。
既存アルゴリズム＝手段、解くこと＝目的

'''

### TLE error
n, w = map(int, input().split())
wlist = [0]*n
vlist = [0]*n
for _ in range(n):
    wlist[i], vlist[i] = list(map(int, input().split()))
dp = [[0 for _  in range(w+1)] for _ in range(n+1)]

for index in range(n):
    for weight in range(w+1):
        if wlist[index] > weight:
            dp[index+1][weight] = dp[index][weight]
        elif wlist[index] <= weight:
            dp[index+1][weight] = max(dp[index][weight],
                                        dp[index][weight - wlist[index]] + vlist[index])

print(dp[n][w])

### reference
# ABC060D - Simple Knapsack (ARC073D)
# exhaustive search
from itertools import accumulate


def main():
    N, W, *A = map(int, open(0).read().split())
    V, x = [[] for _ in range(4)], A[0]  # x: w1 (W1 ≤ wi ≤ w1 + 3)
    for i in range(0, 2 * N, 2):  # group by weight
        w, v = A[i : i + 2]
        V[w - x] += [v]
    for i in range(4):  # V[i][j] := max value of picking up j items from group i
        V[i] = tuple(accumulate([0] + sorted(V[i], reverse=1)))
    L = [len(v) for v in V]
    ans = 0
    for i in range(L[0]):
        for j in range(L[1]):
            for k in range(L[2]):
                w = i * x + j * (x + 1) + k * (x + 2)
                if w > W:
                    break
                l = min(L[3] - 1, (W - w) // (x + 3))
                v = V[0][i] + V[1][j] + V[2][k] + V[3][l]
                ans = max(ans, v)
    print(ans)


if __name__ == "__main__":
    main()

########################################################################################################################################################################
############# ABC061
########################################################################################################################################################################
a, b, c = list(map(int, input().split()))
if (c >= a) and (c <= b):
    print('Yes')
else:
    print('No')



n, m = list(map(int, input().split()))

roads = [[] for _ in range(n)]
for _ in range(m):
    f, t = list(map(int, input().split()))
    roads[f-1].append(t-1)
    roads[t-1].append(f-1)
for i in range(n):
    print(len(roads[i]))

n, k = list(map(int, input().split()))
dic = {}
for _ in range(n):
    a, b = list(map(int, input()))
    if a not in dic:
        dic[a] = b
    else:
        dic[a] += b
total = 0
for key, val in sorted(dic.items(), key=lambda x: x[0]):
    total += val
    if k<= total:
        print(key)
        break

'''
title: Score Attack
keyword: 最短経路問題、ダイクストラ法、ベルマンフォード法、ワーシャフロイド法
study:
&&&ダイクストラ法、ベルマンフォード法、ワーシャフロイド法
'''
#### 分かるか！
from collections import deque
n, m = map(int, input().split())
root = [[] for _ in range(n)]
for _ in range(n):
    f, t, w = map(int, input().split())
    root[f-1].append((t-1, w))
inf = float('inf')
costs = [0 for _ in range(n)]

que = deque()
que.append(0)
while que:
    next = que.popleft()
    for to, weight in root[next]:
        que.append(to)

### 提出7353684
n,m=map(int,input().split())

def bellman_ford(v, s, e):
    INF = 10**18
    # コストをINFで初期化
    d = [INF] * v
    # 開始頂点は0
    d[s] = 0
    # 負の閉路が無ければ更新はV-1回までで終わる
    for i in range(v):
        f = False
        for a, b, c in e:
            if d[a]==INF:continue
            # aまでのコスト+辺abのコストがbまでのコストより小さければ更新
            cost = d[a] + c
            if cost < d[b]:
                d[b] = cost
                f = True
                if i==v-1 and b==v-1:
                    return -1
        # 更新が無ければbreak
        if not f:
            break

    return d
e=[]
for i in range(m):
    a,b,c=[int(j) for j in input().split()]
    e.append((a-1,b-1,-c))

ans=bellman_ford(n,0,e)
if ans==-1:
    print("inf")
else:
    print(-ans[-1])

########################################################################################################################################################################
############# ABC062
########################################################################################################################################################################
x, y = map(int, input().split())

g1 = [1,3,5,7,8,10,12]
g2 = [4,6,9,11]
g3 = [2]
judge = []
for index, g in enumerate([g1, g2, g3]):
    if x in g:
        judge.append(index)
    if y in g:
        judge.append(index)
if len(set(judge)) == 1:
    print('Yes')
else:
    print('No')

h, w = map(int, input().split())

print('#' * (w+2))

for _ in range(h):
    print('#', end='')
    row = list(map(int, input().split())
    print(''.join(row), end='')
    print('#')

print('#' * (w+2))

###### C問題、分からなかった。
def c_chocolate_bar(H, W):
    # 変数a, b, cはeditorialの図のA, B, Cに対応する
    ans = float('inf')
    for _ in range(2):
        for i in range(1, H):
            a = i * W  # 領域Aの面積

            # 横に2つ切り込みを入れる(図の左から1番目の分け方)
            b = ((H - i) // 2) * W
            c = ((H - i + 1) // 2) * W
            ans = min(ans, abs(max(a, b, c) - min(a, b, c)))

            # 横縦に1つずつ切り込みを入れる(図の左から2番目の分け方)
            b = (W // 2) * (H - i)
            c = ((W + 1) // 2) * (H - i)
            ans = min(ans, abs(max(a, b, c) - min(a, b, c)))
        H, W = W, H  # H,Wを入れ替えて同じことをする(図の左から3, 4番目の分け方)
    return ans

H, W = [int(i) for i in input().split()]
print(c_chocolate_bar(H, W))

H, W = map(int, input().split())

ans = float('inf')
for _ in range(2):
    for i in range(1, H):
        a = i*W

        b = ((H - i) // 2) * W
        c = ((H - i + 1)// 2) * W
        ans = min(ans, abs(max(a, b, c) - min(a, b, c)))

        b = (W // 2) * (H -i)
        c = ((W + 1) // 2) * (H - i)
        ans = min(ans, abs(max(a, b, c) - min(a, b, c)))
    H, W = W, H
return ans

### D問題未回答

########################################################################################################################################################################
############# ABC063
########################################################################################################################################################################
a, b = map(int, input().split())
val = a+b
if val >= 10:
    print('error')
else:
    print(val)

slist = list(input())
sset = set(slist)
if len(slist) == len(sset):
    print('yes')
else:
    print('no')


'''
title: Bugged
unkown:条件を満たす(最小値、最大値)
algorithm:
otheridea:
１ステップ言い換えただけでは解けなくて、さらなる言い換えが必要だった
'''
n = int(input())
slist = [int(input()) for _ in range(n)]
total = sum(slist)

if total % 10 != 0:
    print(total)
    exit()
else:
    nonten = []
    for i in range(n):
        if slist[i] % 10 != 0:
            nonten.append(slist[i])
    if len(nonten) == 0:
        print(0)
    else:
        nonten = sorted(nonten)
        total -= nonten[0]
        print(total)

# D問題解けない。
'''
title: Widespread
unkown:条件を満たす(最小値、最大値)
algorithm:二分探索
otheridea:
問題の言い換え、二分探索に帰着させること
・問題を言い換えようとする意志
・二分探索への理解(二分探索の一般化)
'''
### reference
import numpy as np
N, A, B = map(int, input().split())
h = []
for i in range(N):
    h.append(int(input()))
H = np.array(h)

### n回以下の攻撃で、全滅させられるかどうか。
def test(n):
    x = H - n*B
    x[x<0] = 0
    　　　　#np.ceilが結局追加攻撃しなければならないそれぞれのモンスターの攻撃階数
    return np.ceil(x/(A-B)).sum() <= n

## left, right←攻撃階数
left = 0
right = 10**9

while right - left > 1:
    mid = (left + right)//2
    if test(mid):
        right = mid
    else:
        left = mid
ans = right
print(ans)


### myanser ⇒WAくらってる。
import numpy as np
n, a, b = map(int, input().split())
hlist = np.array([int(input()) for _ in range(n)])

def kill(k):
    tmp = hlist - k*b
    tmp[tmp<0] = 0
    if np.ceil(tmp/(a-b)).sum() <= n:
        return True
    else:
        return False

left = 0
right = 10*9
while right - left > 1:
    mid = (right + left)//2
    if kill(mid):
        right = mid
    else:
        left = mid
ans = right
print(ans)


########################################################################################################################################################################
############# ABC064
########################################################################################################################################################################
rgb = list(input().split())
rgb = ''.join(rgb)
rgb = int(rgb)
if rgb % 4 == 0:
    print('YES')
else:
    print('NO')

n = int(input())
alist = list(map(int, input().split()))
dis = max(alist) - min(alist)
print(dis)

'''
title: Colorful Leaderboard
unkown:条件を満たす最小、最大値
algorithm:
otheridea:
条件をしたのように手打ちするのは、きわどいテストケースに必ず引っ掛かるように
作られているので避ける。一般的表現を見つける。
'''
reet = [(1,399),(400,799),(800,1199),(1200,1599),
        (1600,1999),(2000,2399),(2400,2799),(2800,3199)]
n = int(input())
alist = list(map(int, input().split()))

dorei = []
osama = 0
for a in alist:
    osamaflag = True
    for i in range(len(reet)):
       	bottom, top = reet[i]
        if (bottom <= a <= top):
            dorei.append(i)
            osamaflag = False
            break
    if osamaflag:
        osama += 1

min_ = len(set(dorei))
if min_ == 0:
    min_ = 1
max_ = len(set(dorei)) + osama
print(min_, max_)

'''
title: Insertion
unkown:条件を満たす最小の文字列
algorithm:
otheridea:
問題の見方を変える。条件を言い換えること
'''
### D問題解けてない。
n = int(input())
s = input()

slist = []
point = 0
for i in range(n):
    judge1 = s[i] == ')' and s[i+1] == '('
    judge2 = i == n-1
    if judge1 or judge2:
        val = len(s[point:i+1])
        slist.append(val)
        point = i+1

correct = []
for s in slist:
    if s%2 == 0:
        correct.append(s)
    if s%2 == 1:
        correct.append(s+1)
correct = sorted(correct)

ans = ''
for c in correct:
    cnt = c//2
    ans += '('*cnt + ')'*cnt
### reference
time = int(input())
kakko = input()
hidarikakko = 0 #(
migikakko = 0 #)
hidariHuyasu = 0 #左に増やす(の数
migiHuyasu = 0 #右に
#左に増やすのを考える
for i in range(time):
    if kakko[i] == "(":
        hidarikakko += 1
    else:
        if hidarikakko == 0:
            hidariHuyasu += 1
        else:
            hidarikakko -= 1
#右に増やすのを考える
hidarikakko = 0 #(
migikakko = 0 #)
gyaku = range(time)
# print(gyaku)
# gyaku.reverse()
for i in reversed(range(time)):
    if kakko[i] == ")":
        migikakko += 1
    else:
        if migikakko == 0:
            migiHuyasu += 1
        else:
            migikakko -= 1
print("(" * hidariHuyasu + kakko + ")" * migiHuyasu)

# myanswer
n = int(input())
s = input()

left = 0
leftplus = 0
for i in range(len(s)):
    if s[i] == '(':
        left += 1
    else:
        if left == 0:
            leftplus += 1
        else:
            left -= 1

right = 0
rightplus = 0
r = s[::-1]
for i in range(len(reverse)):
    if r[i] == ')':
        right += 1
    else:
        if right == 0:
            rightplus += 1
        else:
            right -= 1
print('('*leftplus + s + ')'*rightplus)

########################################################################################################################################################################
############# ABC065
########################################################################################################################################################################
x, a, b = map(int, input().split())
time = b-a

if time <= 0:
    print('delicious')
elif (0 < time <= x):
    print('safe')
else:
    print('dangerous')

'''
title: Trained?
unkown:条件を満たせるか否か、条件を満たす最小値
algorithm:　シュミレーション
otheridea:
計算時間への配慮
x in listはO(n)の計算時間がかかる。
https://qiita.com/Hironsan/items/68161ee16b1c9d7b25fb
'''
n = int(input())
alist = [int(input())-1 for _ in range(n)]

now = 0
count = 0
while True:
    if now == 1:
        print(count)
        break
    elif count >= n:
        print(-1)
        break
    else:
        count += 1
        now = alist[now]

'''
title: Trained?
unkown:条件を満たせるか否か、条件を満たす最小値
algorithm:　シュミレーション
otheridea:
計算時間への配慮
x in listはO(n)の計算時間がかかる。
https://qiita.com/Hironsan/items/68161ee16b1c9d7b25fb
'''
import math
def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))
mod = 10**9 + 7
n, m = map(int, input().split())
diff = max(n, m) - min(n, m)

if diff >=  2:
    print(0)
    exit()

inup = math.factorial(n)
sarup = math.factorial(m)
if diff == 0:
    val = inup * sarup * 2
    print(val%mod)
elif diff == 1:
    val = inup * sarup
    print(val%mod)

'''
title: Builted?
unkown:最小全域木問題
algorithm:　クラスカル法、プリム法
otheridea:
'''
# D問題参考
import sys
sys.setrecursionlimit(10**7)
def LI(): return [int(x) for x in sys.stdin.readline().split()]
def LI_(): return [int(x) - 1 for x in sys.stdin.readline().split()]
def LF(): return [float(x) for x in sys.stdin.readline().split()]
def LS(): return sys.stdin.readline().split()
def II(): return int(sys.stdin.readline())
def SI(): return sys.stdin.readline().strip()
INF = 10 ** 18
MOD = 10 ** 9 + 7


import heapq
def prim_heap(n, edges):  # n: 最大の頂点番号+1, edges[頂点番号]=[[重み, 行き先],[重み, 行き先],...]. 無向グラフの場合は相方向有向グラフの表記にしておく。
    seen = [False] * n #True:不使用
    edgelist = []  # heapq
    for e in edges[0]:  # 頂点 0 から伸びる辺をすべて heapq に入れる。
        heapq.heappush(edgelist,e)
    seen[0] = True  # 頂点 0 は見たと記録する。
    res = 0
    while len(edgelist) != 0:
        minedge = heapq.heappop(edgelist)  # edgelist から最小の辺を1つ取り出す。
        if seen[minedge[1]]:  # その辺の行き先がすでに見た先だったら棄却して次へ。
            continue
        v = minedge[1]  # 行き先を v とする。
        seen[v] = True  # 頂点 v は見たと記録する。
        for e in edges[v]:  # 頂点 v から伸びる辺すべてについて、もしその行き先を見ていなかったら heapq に追加する。
            if not seen[e[1]]:
                heapq.heappush(edgelist,e)
        res += minedge[0]  # いま使った辺の重みを結果に加える。
    return res  # 最小全域木の全体の重みを出力する。



def main():
    N = II()
    pos_li = []
    for i in range(N):
        x,y = LI()
        pos_li.append((x,y,i))
    edges = [[] for _ in range(N)]
    # # すべての頂点を結ぶグラフを作ってその最小全域木を取ると、時間が足りない（二重ループが O(10^10)）。
    # for i in range(N):
    #     for j in range(N):
    #         weight = min(abs(pos_li[i][0] - pos_li[j][0]), abs(pos_li[i][1] - pos_li[j][1]))
    #         edges[i].append([weight, j])
    #         edges[j].append([weight, i])
    # # なので、座標でソートして、最も近い4点のみへ辺を伸ばす。
    pos_li.sort(key=lambda x: x[0])
    for k in range(N-1):
        x1, y1, p1 = pos_li[k]
        x2, y2, p2 = pos_li[k+1]
        weight = abs(x1-x2)
        edges[p1].append([weight, p2])
        edges[p2].append([weight, p1])

    pos_li.sort(key=lambda x: x[1])
    for k in range(N-1):
        x1, y1, p1 = pos_li[k]
        x2, y2, p2 = pos_li[k+1]
        weight = abs(y1-y2)
        edges[p1].append([weight, p2])
        edges[p2].append([weight, p1])

    # print(edges)
    print(prim_heap(N, edges))

########################################################################################################################################################################
############# ABC141
########################################################################################################################################################################
s = input()
if s == 'Sunny':
    print('Cloudy')
if s == 'Cloudy':
    print('Rainy')
if s == 'Rainy':
    print('Sunny')

st = input()
flag = 'Yes'
for i in range(len(st)):
    if i % 2 == 1:
        if st[i] not in ['R', 'U', 'D']:
            flag = 'No'
    else:
        if st[i] not in ['L', 'U', 'D']:
            flag = 'No'
print(flag)


from collections import Counter
n, k , q = map(int, input().split())
qlist = [int(input()) for _ in range(q)]
count = Counter(qlist)

for i in range(1, n+1):
    damage = q - count[i]
    if damage < k:
        print('Yes')
    elif damage >= k:
        print('No')
'''
title: Builted?
unkown:最小全域木問題
algorithm:　クラスカル法、プリム法
otheridea:
'''
# D 問題解けなかった。
# 貪欲法的考え方はあったが、priority queの知識がなく、計算量の壁が乗り越えられなかった。
n, m = map(int, input().split())
alist = [int(input) for _ in range(n)]
alist = np.array(alist)
alist = np.sort(alist)[::-1]
border = min(alist)
while m != 0:
    tr = sum(alist > border)
    if tr <= m:
        alist = np.where(alist > border, alist//2, alist)
        m -= tr
    if tr > m:
        alist = np.sort(alist)[::-1]
        for i in range(len(alist)):
            alist[i] = alist[i]//2
        break

    alist = np.sort(alist)[::-1]
print(sum(alist))

# 参考　python だとheapqという名前でpriorityqueの実装がなされているらしい。
# https://juppy.hatenablog.com/entry/2018/11/08/%E8%9F%BB%E6%9C%AC_python_%E3%83%97%E3%83%A9%E3%82%A4%E3%82%AA%E3%83%AA%E3%83%86%E3%82%A3%E3%82%AD%E3%83%A5%E3%83%BC_heapq_%E7%AB%B6%E6%8A%80%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3
# https://qiita.com/ellio08/items/fe52a9eb9499b7060ed6
from heapq import heappop,heappush,heapify

N, M = map(int, input().split())
A = list(map(int, input().split()))
A_minus = [-a for a in A]

heapify(A_minus)

for _ in range(M):
    heappush(A_minus, -((-heappop(A_minus)) // 2))
ans = -sum(A_minus)
print(int(ans))

# myanswer
from heapq import heapify, heappop, heappush
n, m = map(int, input().split())
al = list(map(int, input().split()))
al = [-a for a in al]
heapify(al)

while m != 0:
    tmp = heappop(al)
    tmp = int(tmp/2)
    m -= 1
    heappush(al, tmp)

al = list(al)
al = [-a for a in al]
print(sum(al))

## E問題解けてない
# 蟻本p56 共通部分和問題。
def main():
    N = int(input())
    S = input()
    ans = 0

    """
    他言語ならこれで行けるはずだが、pythonだとTLE
    dp = [[0 for _ in range(N+10)] for _ in range(N+10)]
    for i in reversed(range(N)):
        for j in reversed(range(i+1,N)):
            if S[i] != S[j]:
                dp[i][j] = 0
            else:
                dp[i][j] = dp[i+1][j+1] + 1
            if i >= j: continue
            tmp = min(dp[i][j], j-i)
            ans = max(ans, tmp)
    """
    i = j = 0

    while  j <= N-1:
        # 頭から見ていき、部分文字列が、その後ろのどこかにあれば範囲を広げる。かぶってはいけないのでj-iが最大
        if S[i:j] in S[j:]:
            ans = max(ans, j-i)
            j += 1
        else:
            i += 1


    print(ans)

if __name__ == "__main__":
    main()

# myanswer
n = int(intput())
s = int(intput())

ans = 0
i = j= 0

while j <= n-1:
    if s[i:j] in s[j:]:
        ans = max(ans, j-i)
        j += 1
    else:
        i += 1
print(ans)

# F問題解けてない。

########################################################################################################################################################################
############# ABC065
########################################################################################################################################################################
li = list(map(int, input().split()))
li = sorted(li)
print(li[0] + li[1])

s = input()
flag = False
ans = 0
for i in range(len(s)-2, 0, -2):
    if s[:i//2] == s[i//2+1:i]:
        flag = True
    if flag:
        ans = i
        break
print(i)

from collections import deque
b = deque(list())
n = int(input())
alist = list(map(int,input().split()))
num = len(alist)

if num % 2 == 0:
    for i in range(num):
        if i % 2 == 0:
            b.append(alist[i])
        elif i % 2 == 1:
            b.appendleft(alist[i])

elif num % 2 == 1:
    for i in range(num):
        if i % 2 == 1:
            b.append(alist[i])
        elif i % 2 == 0:
            b.appendleft(alist[i])

for i in b:
    print(i, end=' ')

### TLE
from math import factorial
from collections import Counter
mod = 10**9 + 7

n = int(input())
length = n+1
li = list(map(int,input().split()))

counter = Counter(li)
dup = counter.most_common()[0][0]
be, af = [i for i, x in enumerate(li) if x==dup]
soto = be + length - af - 1

for k in range(1, n+2):
    if k == 1:
        print(n)
    elif k == n+1:
        print(1)
    else:
        zyouhukuari = factorial(length) // (factorial(k)*factorial(length-k))
        if soto-k+1 >= 0:
            zyouhuku = factorial(soto) // (factorial(k-1)*factorial(soto-k+1))
        else:
            zyouhuku = 0
        val = zyouhukuari - zyouhuku
        print(val%mod)

### reference
import collections

N = int(input())
a = list(map(int, input().split()))

c = collections.Counter(a)
dup = c.most_common()[0][0]

id = [i for i, x in enumerate(a) if x == dup]
L = id[0]
R = id[1]

MOD = 10**9 +7
factorial = [1]
inverse = [1]

for i in range(1, N+2):
    factorial.append(factorial[i-1] * i % MOD)
    inverse.append(pow(factorial[i], MOD-2, MOD))


def C(n, r):
    if n < r:
        return 0
    return (factorial[n] * inverse[r] * inverse[n-r]) % MOD

print(N)
for i in range(2, N+2):
    ans = (C(N+1, i) - C(L+(N-R), i-1)) % MOD
    print(ans)

########################################################################################################################################################################
############# ABC067
########################################################################################################################################################################
a, b = map(int, input().split())
data = [a, b, a+b]
flag = 'Impossible'
for d in data:
    if d % 3 == 0:
        flag = 'Possible'
print(flag)

n, k = map(int, input().split())
li = list(map(int, input().split()))
li = sorted(li, reverse=True)
maxtotal = 0
for i in range(k):
    maxtotal += li[i]
print(maxtotal)

## 解けなかった。⇒問題の言い換えが足らない！
import numpy as np
n = int(input())
li = map(int, input().split())

be = 0
af = 0
for i, d in enumerate(li):
    af += d
    if af >= mean:
        bediff = abs(mean - be)
        afdiff = abs(mean - af)
        if bediff <= afdiff:
            maetotal = be
            point = i-1
        elif bediff > afdiff:
            maetotal = af
            point = i
        break
    be = af
print(maetotal - sum(li[point+1:])
###
n = int(input())
li = list(map(int, input().split()))
total = sum(li)

sum_ = 0
min_ = float('inf')
for i in range(n-1):
    sum_ += li[i]
    diff = abs(total - 2*sum_)
    if diff < min_:
        min_ = diff
print(min_)

### D問題
from collections import deque
n = int(input())
g = [[]*n]
for _ in range(n):
    a, b = map(int, input().split())
    g[a-1] = b-1
    g[b-1] = a-1
used = [0] * n
fq = deque(list())
sq = deque(list())
find = 0
sind = n-1
while True:
    used[find] = 1
    choice = g[find]

########################################################################################################################################################################
############# ABC068
########################################################################################################################################################################
n = input()
print('ABC{:0>3}'.format(n))

n = int(input())

if n == 1:
    print(1)
    exit()

for i in range(1, 8):
    if 2**i <= n < 2**(i+1):
        ans = 2**i
print(ans)

n, m = map(int, input().split())
edge = [[] for _ in range(n)]
for _ in range(m):
    a, b = map(int, input().split())
    edge[a-1] = b-1
    edge[b-1] = a-1

from0 = edge[0]
flag = 'IMPOSSIBLE'
for i in from0:
    if n-1 in edge[i]:
        flag = 'POSSIBLE'
print(flag)

###  D問題reference
K = int(input())
N = 50

ans = list(range(N))

for i in range(len(ans)):
    ans[i] += K//N

for i in range(K % N):
    for j in range(len(ans)):
        if j == i:
            ans[j] += N
        else:
            ans[j] -= 1

print(N)
print(*ans)

## D問題myans
k = int(input())
n= 50
lastli = list(range(n))

syo = k//n
amari = k%n

for i in range(len(lastli)):
    lastli[i] += syo

for i in range(len(amari)):
    lastli[i] += 1

print(n)
print(*lastli)

########################################################################################################################################################################
############# ABC069
########################################################################################################################################################################
n, m = map(int, input().split())
print((n-1)*(m-1))

s = input()
print(s[0] + str(len(s[1:-1])) + s[-1])

# WA
####  (%4 == 2) と (%2 == 0) は、別物なので注意！
n = int(input())
li = list(map(int, input().split()))
yon = 0
ni  = 0
ki = 0
for a in li:
    if a % 4 == 0:
        yon += 1
    if a % 4 == 2:
        ni += 1
    else:
        ki += 1
if ki == 0:
    print('Yes')
    exit()

if ni > 0:
    if yon >= ki:
        print('Yes')
    else:
        print('No')
elif ni == 0:
    if yon >= ki-1:
        print('Yes')
    else:
        print('No')

# revised myanswer
n = int(input())
li = list(map(int, input().split()))
yon = 0
ni  = 0
ki = 0
for a in li:
    if a % 4 == 0:
        yon += 1
        continue
    if a % 2 == 0:
        ni += 1
    else:
        ki += 1

if ni > 0:
    if yon >= ki:
        print('Yes')
    else:
        print('No')
elif ni == 0:
    if yon >= ki-1:
        print('Yes')
    else:
        print('No')

# D問題
inf = float('inf')

h, w = map(int, input().split())
n = int(input())
li = list(map(int, input().split()))
fi = [[inf for _ in range(w)] for _ in range(h)]

x = 0
y = 0

for color, count in enumerate(li):
     color = color + 1
     while count != 0:
        fi[x][y] = color
        count -= 1

        if x % 2 == 0:
            nexty = y+1
        else:
            nexty = y-1

        if nexty == 0 or nexty == W:
            x = x+1
        elif fi[x][nexty] == inf:
            y = nexty

for row in fi:
    print(*row)

########################################################################################################################################################################
############# ABC070
########################################################################################################################################################################
n = input()
nr = n[::-1]
if n == nr:
    print('Yes')
else:
    print('No')

a, b, c, d = map(int, input().split())
start = max(a, c)
stop = min(b, d)
val = stop - start

if val <= 0:
    print(0)
elif val > 0:
    print(val)

import math
from functools import reduce
def lcm_base(x, y):
    return (x * y) // math.gcd(x, y)

def lcm(*numbers):
    return reduce(lcm_base, numbers, 1)

def lcm_list(numbers):
    return reduce(lcm_base, numbers, 1)

n = int(input())
li = [int(input()) for _ in range(n)]

print(lcm_list(li))


def gcd(x, y):
    while y > 0:
        x, y = y, x%y
    return x
def lcm(x, y):
    return x/gcd(x, y)*y

n = int(input())
val = int(input())
for i in range(n-1):
    tmp = int(input())
    val = int(lcm(val, tmp))

print(val)

# D Transit Tree Path
inf = float('inf')

n = int(input())
g = [[] for _ in range(n)]
used = [False] * n
dis = [inf] * n

for _ in range(n):
    a, b, weight = map(int, input().split())
    g[a-1].append((b-1, weight))
    g[b-1].append((a-1, weight))

while True:
    v = -1
    min_ = inf
    for u in range(n):
        if (not used[u]) and (dis[u] < min_):
            v = u
            min_ = dis[u]

    if v == -1:
        break

    used[v] = True

    for i in range(n):
        d[i] = min(d[i], d[v] + cost)


########################################################################################################################################################################
############# ABC071
########################################################################################################################################################################

# A
x, a, b = map(int, input().split())
da = abs(x-a)
db = abs(x-b)
if da < db:
    print('A')
elif db < da:
    print('B')

# B
import string
alpha = list(string.ascii_lowercase)
s = input()
li = [st for st in alpha if st not in s]
if len(li) == 0:
    print('None')
elif len(li) > 0:
    print(li[0])
# C
from collections import Counter
n = int(input())
li = list(map(int, input().split()))
counter = Counter(li)
dic = {}
for key, val in counter.items():
    if val >= 2:
        dic[key] = val

if len(dic) == 0:
    print(0)
elif len(dic) == 1:
    key, val = dic.popitem()
    if val >= 4:
        print(key*key)
    if val <= 3:
        print(0)
elif len(dic) >= 2:
    sort = sorted(dic.items(), key=lambda x:x[0], reverse=True)
    k1, v1 = sort[0]
    k2, v2 = sort[1]
    if v1 >= 4:
        print(k1*k1)
    elif v1 <= 3:
        print(k1*k2)


# D
import numpy as np
n = int(input())
s1 = list(input())
s2 = list(input())

domino = np.array([s1, s2])
array = []
point = 0
while True:
    col = domino[:, point]
    if col[0] == col[1]:
        array.append(0) # X
        point += 1
    else:
        array.append(1) # Y
        point += 2
    if point >= n:
        break

total = 0
for i in range(len(array)):
    if i == 0:
        if array[i] == 0:
            total += 3
        else:
            total += 6
    else:
        if array[i] == 0:
            if array[i-1] == 0:
                total += 2
            else:
                total += 1
        else:
            if array[i-1] == 0:
                total += 2
            else:
                total += 3
print(total % (10**9 + 7))

########################################################################################################################################################################
############# ABC072
########################################################################################################################################################################

# A
x, t = map(int, input().split())
val = x - t
if val <= 0:
    print(0)
else:
    print(val)

# B
st = input()
ans = [s for ind, s in enumerate(st) if ind % 2 == 0]
ans = ''.join(ans)
print(ans)

# C
from collections import Counter
n = int(input())
ban = []
for a in map(int, input().split():
    ban.append(a)
    ban.append(a-1)
    ban.append(a+1)
counter = Counter(ban)
most = counter.most_common()[0]
print(most[1])

# D
n = int(input())
ps = list(map(int, input().split()))
sikori = []
for ind, p in enumerate(ps):
    point = ind+1
    if p[ind] == point:
        sikori.append(point)

if len(sikori) == 0:
    print(0)
elif len(sikori) == 1:
    print(1)
else:
    count = 0
    loc = 0
    while loc != len(sikori)-1:
        if shikori[loc] + 1 == shikori[loc+1]:
            count += 1
            loc += 2
        else:
            count += 1
            loc += 1
    print(count)

########################################################################################################################################################################
############# ABC073
########################################################################################################################################################################

# A
s = list(input())
if '9' in s:
    print('Yes')
else:
    print('No')
# B
n = int(input())
total = 0
for _ in range(n):
    l, r = map(int, input().split())
    val = (r+1) - l
    total += val
print(total)

# C
from collections import Counter
n = int(input())
ali = [int(input()) for _ in range(n)]

counter = Counter(ali)
count = 0
for k, v in counter.items():
    if v % 2 == 1:
        count += 1
print(count)

# D

########################################################################################################################################################################
############# ABC074
########################################################################################################################################################################

# A
n = int(input())
siro = int(input())
zen = n*n
kuro = zen - siro
if kuro <= 0:
    print(0)
else:
    print(kuro)


# B
n = int(input())
k = int(input())
xs = list(map(int, input().split()))

min_list = []
for x in xs:
    min_ = min(abs(x), abs(x-k))
    min_list.apend(min_)

total = 0
for m in min_list:
    total += 2*m
print(total)


# C
a, b, c, d, e, f = list(map(int, input().split()))
def checklist(list, f):
    if list[-1]>f:
        break

wali = []
i = 0
j = 0
while True:
    wali.append(100*a*(i+1) + 100*b*j)
    checklist(wali, f)
    wali.append(100*a*i + 100*b*(j+1))
    checklist(wali, f)
    wali.append(100*a*(i+1) + 100*b*(j+1))
    checklist(wali, f)
    i += 1
    j += 1

suli = []
i = 0
j = 0
while True:
    suli.append(c*(i+1) + d*j)
    checklist(suli, f)
    suli.append(c*i + d*(j+1))
    checklist(suli, f)
    suli.append(c*(i+1) + d*(j+1))
    checklist(suli, f)
    i += 1
    j += 1


lc = lcm(c, d)



# D


########################################################################################################################################################################
############# ABC142
########################################################################################################################################################################

# A
n = int(input())
total = 0
for i in range(1, n+1):
    if i % 2 == 1:
        total += 1
print(total/n)

# B
n, k = map(int, input())
hs = list(map(int, input().split()))
total = 0
for h in hs:
    if h >= k:
        total += 1
print(total)

# C
n = int(input())
ali = list(map(int, input().split()))
kumi = []
for ind, a in enumerate(ali, start=1):
    kumi.append((ind, a))
sorted_kumi = sorted(kumi, key=lambda x:x[1])
for ind, a in sorted_kumi:
    print(ind, end=' ')

# D
import fractions
from itertools import combinations
import heapq
def cf(x1,x2):
    cf=[1]
    for i in range(2,min(x1,x2)+1):
        if x1 % i == 0 and x2 % i == 0:
            cf.append(i)
    return cf
def compromise(x1, x2):
    f = fractions.gcd(x1, x2)
    return f == 1
def prime_factorize(n):
    a = []
    while n % 2 == 0:
        a.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            a.append(f)
            n //= f
        else:
            f += 2
    if n != 1:
        a.append(n)
    return a
def isPrime(n):
  if n < 2:
    # 2未満は素数でない
    return False
  if n == 2:
    # 2は素数
    return True
  for p in range(2, n):
      if n % p == 0:
        # nまでの数で割り切れたら素数ではない
        return False
  # nまでの数で割り切れなかったら素数
  return True

a, b = map(int, input().split())
cfs = cf(a, b)
apri = set(prime_factorize(a))
bpri = set(prime_factorize(b))
print(len(apri & bpri)+1)


cfs = cf(a, b)
total = len(cfs)
for i in range(len(cfs)):
    if i == 0:
        continue
    for j in range(i+1, len(cfs)):
        if cfs[j] % cfs[i] == 0:
            total -= 1
print(total)


total = 0
for i in range(len(cfs)):
    if i == 0:
        total += 1
    else:
        flag = True
        for c in club:
            if not compromise(cfs[i], c):
                flag = False
                break
        if flag:
            total += 1
print(len(club))



# E

# F


########################################################################################################################################################################
############# ABC075
########################################################################################################################################################################

# A
from collections import Counter
li = list(map(int, input().split()))
count = Counter(li)
print(count.most_common()[1][0])

# B
import numpy as np
h, w = map(int, input().split())
fi = np.array([['.' for _ in range(w)] for _ in range(h)])
for i in range(h):
    si = input()
    for j, s in enumerate(si):
        fi[i, j] = s
xs = [-1, 0, 1]
ys = [-1, 0, 1]
for i in range(h):
    for j in range(w):
        if fi[i, j] == '#':
            continue
        else:
            cnt = 0
            for x in xs:
                for y in ys:
                ni = i + x
                nj = j + y
                if (0 <= ni <= h-1) and (0 <= nj <= w-1):
                    if fi[ni, nj] == '#':
                        cnt += 1
            fi[i, j] = str(cnt)
for row in fi:
    print(''.join(row))


# C
from collections import deque

def Connected(g, nodenum):
    visited = [False]*nodenum
    que = deque([])
    posi = 0
    que.append(posi)
    while que:
        visit = que.pop()
        visited[visit] = True
        for goto in g[visit]:
            if visited[goto]:
                continue
            que.append(goto)
    if all(visited):
        return True
    else:
        return False

n, m = map(int, input().split())
g = [[] for _ in range(n)]
edge_log = []
for _ in range(m):
    a, b = map(int, input().split())
    g[a-1].append(b-1)
    g[b-1].append(a-1)
    edge_log.append((a-1, b-1))

ans = 0
for edge in edge_log:
    g[edge[0]].remove(edge[1])
    g[edge[1]].remove(edge[0])
    if not Connected(g, nodenum=n):
        ans += 1
    g[edge[0]].append(edge[1])
    g[edge[1]].append(edge[0])
print(ans)


n, k = map(int,input().split())
xs = []
ys = []
points = []
for _ in range(n):
    x, y = map(int, input().split())
    xs.append(x)
    ys.append(y)
    points.append((x, y))
xs = list(sorted(xs))
ys = list(sorted(ys))
min_ = float('inf')
for i in range(len(xs)):
    for j in range(i+1, len(xs)):
            if i == j:
                continue
        for k in range(len(ys)):
            for l in range(k+1, len(ys)):
                if k == l:
                    continue
                cnt = 0
                for tx, ty in points:
                    if (xs[i] <= tx <= xs[j]) and (ys[k] <= ty <= ys[l]):
                        cnt += 1
                if cnt >= k:
                    yoko = xs[j] - xs[i]
                    tate = ys[l] - ys[k]
                    men = yoko*tate
                else:
                    continue
                if men <= min_:
                    min_ = men
print(min_)


min_ = float('inf')
for leftx in xs:
    for rightx in xs:
        if leftx >= rightx:
            continue
        for downy in ys:
            for upy in ys:
                if downy >= upy:
                    continue
                cnt = 0
                for tx, ty in points:
                    if (leftx <= tx <= rightx) and (downy <= ty <= upy):
                        cnt += 1

                if cnt >= k:
                    yoko = rightx - leftx
                    tate = upy - downy
                    men = yoko*tate
                    if men < min_:
                        min_ = men
print(min_)

### reference
#https://atcoder.jp/contests/abc075/submissions/2242350
N,K=map(int,input().split())

points=[tuple(map(int,input().split())) for i in range(N)]
answer=10**19
sortx=sorted(points)
numberx = list(enumerate(sortx))

#n-k+1以降を左の頂点とすると点の数が明らかに足りない
for left,(x1,y1) in numberx[:N-K+1]:
  #left+K-1以降を右の頂点にしないと間に足りる点の数が入ってない
  for right,(x2,y2) in numberx[left+K-1:]:
    dx=x2-x1
    #left-rightの間でyについてsort
    sorty=sorted(y for x,y in sortx[left:right+1])
    #K個点を入れた状態で面積を求める
    for y3,y4 in zip(sorty,sorty[K-1:]):
      if y3<=y1 and y3<=y2 and y4>=y1 and y4>=y2:
        answer=min(answer,dx*(y4-y3))

print(answer)



########################################################################################################################################################################
############# ABC076
########################################################################################################################################################################
# A
r, g = map(int, open(0).read().split())
print(2*g-r)

# B
n = int(input())
k = int(input())

now = 1
for _ in range(n):
    now = min(now*2, now+k)
print(now)

# C
sd = input()
scnt = len(sd)
t = input()
tcnt = len(t)

point = 0
unrestorable = True
while point <= scnt - tcnt:
    oneroopcorres = True
    for i in range(tcnt):
        anyone = sd[point+i] == '?'
        correspond = sd[point+i] == t[i]
        if not (anyone or correspond):
            oneroopcorres = False

    if oneroopcorres:
        unrestorable = False
        good_point = point

    point += 1

if unrestorable:
    print('UNRESTORABLE')
    exit()
else:
    sd = list(sd)
    for i in range(tcnt):
        sd[good_point + i] = t[i]
    sd = ''.join(sd)

sd = sd.replace('?', 'a')
print(sd)

# D

### 解けないーーーー。。

########################################################################################################################################################################
############# ABC077
########################################################################################################################################################################
# A
c1, c2, c3 = input().split()
c4, c5, c6 = input().split()

flag = True
for x, y in [(c1, c6), (c3, c4), (c2, c5)]:
    if x != y:
        flag = False
if flag:
    print('YES')
else:
    print('NO')

# B
n = int(input())

ind = 1
while next <= n:
    now = ind**2
    next = (ind+1)**2
    ind += 1
print(now)
# C
import bisect
n = int(input())
al = map(int, input().split())
bl = map(int, input().split())
cl = map(int, input().split())
al = sorted(al)
bl = sorted(bl)
cl = sorted(cl)

total = 0
for blnum in bl:
    alcnt = bisect.bisect_left(al, blnum)
    clcnt = n - bisect.bisect_right(cl, blnum)
    total += alcnt*clcnt
print(total)

# D
k = int(input())
nodes = list(range(k))
g = [[] for _ in range(k)]

for i in range(k):
    if i == k-1:
        g[i].append(0)
    else:
        g[i].append(i+1)

for i in range(k):
    val = (i*10) % k
    g[i].append(val)

### reference
from collections import deque

def bfs(K:int) -> list:
    hasChecked = [False]*(K+1)
    cost_list = [0]*(K+1)
    cost_list[1] = 1

    que = deque([(cost_list[1], 1)])

    while que:
        cost, res = que.popleft()
        if hasChecked[res]:
            continue

        hasChecked[res] = True
        cost_list[res] = cost

        if not hasChecked[10*res%K]:
            que.appendleft((cost, 10*res%K))

        if not hasChecked[(res+1)%K]:
            que.append((cost+1, (res+1)%K))


    return cost_list


K = int(input())
cost_list = bfs(K)
print(cost_list[0])

########################################################################################################################################################################
############# ABC078
########################################################################################################################################################################
# A
li = list(input().split())
map_ = {'A':10 , 'B':11, 'C':12, 'D':13, 'E':14, 'F':15}
intli = [map_[i] for i in li]

if intli[0]<intli[1]:
    print('<')
elif intli[0]>intli[1]:
    print('>')
else:
    print('=')


# B
x, y, z = map(int, input().split())

cnt = 0
cm = z
while (cm < x):
    cm += y
    if (cm+z) > x:
        break
    elif (cm+z)<=x:
        cm += z
        cnt += 1
print(cnt)


# C
n, m = map(int, input().split())
x = 1900*m + 100(n-m)
p = (0.5)**m
print(x//p)

# D
n, z, w = map(int, input().split())
ali = list(map(int, input().split()))

## dp
N, Z, W = map(int, input().split())
a = list(map(int, input().split()))
dp1 = [0]*N
dp2 = [0]*N
dp1[N-2] = dp2[N-2] = abs(a[N-2]-a[-1])
for i in range(N-2)[::-1]:
    dp1[i] = max(abs(a[i]-a[-1]), max(dp2[j] for j in range(i+1, N-1)))
    dp2[i] = min(abs(a[i]-a[-1]), min(dp1[j] for j in range(i+1, N-1)))
if N==1:
    ans = abs(W-a[-1])
else:
    ans = max(abs(W-a[-1]), max(dp2[i] for i in range(N-1)))
print(ans)

########################################################################################################################################################################
############# ABC078
########################################################################################################################################################################
# A
s = input()
j1 = (s[0] == s[1]) and (s[1]==s[2])
j2 = (s[1] == s[2]) and (s[2]==s[3])
if j1 or j2:
    print('Yes')
else:
    print('No')

# B
def ryuka(n):
    if n == 0:
        return 2
    if n == 1:
        return 1
    return ryuka(n-1) + ryuka(n-2)

n = int(input())
print(ryuka(n))
# C
li = list(input())
li = [int(num) for num in li]

def dfs(index, total, ans):
    print(index)
    if total == 7:
        return ans

    if index == 0:
        total += li[index]
        ans += str(li[index])
        dfs(index + 1, total, ans)
    else:
        dfs(index + 1, total + li[index], ans + '+' + str(li[index]))
        dfs(index + 1, total + li[index], ans + '-' + str(li[index]))

print(dfs(0, 0, '') + '=7')
# D
from collections import Counter
from collections import deque
h, w = map(int, input().split())

g = [[] for _ in range(10)]
for i in range(10):
    row = map(int, input().split())
    for ind, v in enumerate(row):
        if ind == i:
            g[i].append((ind, float('inf')))
        else:
            g[i].append((ind, v))

nums = []
for _ in range(h):
    arow = list(map(int, input().split()))
    nums += arow
counter = Counter(nums)
print(counter)
print(g)

### dijkstra
def dijkstra(s,n,w,cost):
    #始点sから各頂点への最短距離
    #n:頂点数,　w:辺の数, cost[u][v] : 辺uvのコスト(存在しないときはinf)
    d = [float("inf")] * n
    used = [False] * n
    d[s] = 0

    while True:
        v = -1
        #まだ使われてない頂点の中から最小の距離のものを探す
        for i in range(n):
            if (not used[i]) and (v == -1):
               v = i
            elif (not used[i]) and d[i] < d[v]:
                v = i
        if v == -1:
               break
        used[v] = True

        for j in range(n):
               d[j] = min(d[j],d[v]+cost[v][j])
    return d

total = 0
for num, cnt in counter.items():
    if (num == -1) or (num == 1):
    dist = dijkstra(s=num, n=10, w=10*10, g)
    total += dist[1]*cnt
return total

###
def mybellman():

def dijkstra


########################################################################################################################################################################
############# ABC080
########################################################################################################################################################################
# A
n, a, b = map(int, input().split())
print(min(b, a*n))
# B
x = input()
sx = int(x)
sum_ = sum([int(num) for num in list(x)])
if sx%sum_ == 0:
    print('Yes')
else:
    print('No')

# C
import numpy as np
n = int(input())
shops = []
for _ in range(n):
    shops.append(list(map(int, input().split())))
scores = []
for _ in range(n):
    scores.append(list(map(int, input().split())))

cans = []
def dfs(ind, array):
    if ind == 10:
        cans.append(array)
    else:
        dfs(ind+1, array + [0])
        dfs(ind+1, array + [1])
dfs(0, [])
cans.remove([0 for _ in range(10)])

totalscorelist = []
for one in cans:
    indexlist = []
    for other in shops:
        one = np.array(one)
        other = np.array(other)
        k = np.dot(one, other)
        indexlist.append(k)
    totalscore = 0
    for scorebord, index in zip(scores, indexlist):
        totalscore += scorebord[index]
    totalscorelist.append(totalscore)
print(max(totalscorelist))

# D

########################################################################################################################################################################
############# ABC081
########################################################################################################################################################################
# A
from collections import Counter
li = list(input())
total = 0
for s in li:
    if s == '1':
        total += 1
print(total)


# B
from collections import Counter
def prime_factorize(n):
    a = []
    while n % 2 == 0:
        a.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            a.append(f)
            n //= f
        else:
            f += 2
    if n != 1:
        a.append(n)
    return a
n = int(input())
li = list(map(int, input().split()))

count2 = []
for i in range(n):
    fact = prime_factorize(li[i]):
    count = Counter(fact)
    try:
        count2.append(count[2])
    except:
        print(0)
        exit()
print(min(count2))

# C
from collections import Counter
n, k = map(int, input().split())
ali = list(map(int, input().split()))

count = Counter(ali)
sortcount = sorted(count.items(), key=lambda x:x[1])

diverse = len(count)
erase = diverse - k
if erase <= 0:
    print(0)
    exit()

num = 0
for key, val in sortcount[:erase]:
    num += val
print(num)

# D
n = int(input())
ali = list(map(int, input().split()))

log = []
max_ = max(ali)
min_ = min(ali)

if abs(max_) >= abs(min_):
    max_index = ali.index(max_)+1
    for i in range(n):
        log_i = i+1
        log.append(max_index, log_i)
    for j in range(n-1):
        log_j = j+1
        mae = log_j
        ushiro = log_j+1
        log.append(mae, ushiro)
elif abs(max_) < abs(min_):
    min_index = ali.index(min_) + 1
    for i in range(n):
        log_i = i + 1
        log.append(min_index, log_i)
    for j in range(1, n)[::-1]:
        log_j = j+1
        mae = log_j
        ushiro = log_j -1
        log.append(mae, ushiro)

print(len(log))
for v1, v2 in log:
    print(str(v1) + ' ' + str(v2))

########################################################################################################################################################################
############# ABC082
########################################################################################################################################################################
# A
import math
a, b = map(int, input().split())
print(math.ceil((a+b)/2))

# B
s = input()
t = input()
sorteds = list(sorted(list(s)))
sortedt = list(sorted(list(t), reverse=True))
s_ = ''.join(sorteds)
t_ = ''.join(sortedt)
if (s_ < t_):
    print('Yes')
else:
    print('No')

# C
from collections import Counter
n = int(input())
al = list(map(int, input().split()))
counter = Counter(al)
total = 0
for key, val in counter.items():
    if key > val:
        total += val
    elif key < val:
        total += val-key
print(total)

# D
# D
### TLE
import sys
sys.setrecursionlimit(5000)
s = input()
x, y = map(int, input().split())
finished = []

def dfs(now, index, direction):
    if index == len(s)-1:
        if s[index] == 'T':
            finished.append(now)
            return
        elif s[index] == 'F':
            if direction==1:
                nowx, nowy = now
                next = (nowx+1, nowy)
                finished.append(next)
                return
            elif direction==2:
                nowx, nowy = now
                next = (nowx, nowy+1)
                finished.append(next)
                return
            elif direction==3:
                nowx, nowy = now
                next = (nowx-1, nowy)
                finished.append(next)
                return
            elif direction==0:
                nowx, nowy = now
                next = (nowx, nowy-1)
                finished.append(next)
                return

    if s[index] == 'F':
        if direction==1:
            nowx, nowy = now
            next = (nowx+1, nowy)
            dfs(next, index+1, direction)

        elif direction==2:
            nowx, nowy = now
            next = (nowx, nowy+1)
            dfs(next, index+1, direction)

        elif direction==3:
            nowx, nowy = now
            next = (nowx-1, nowy)
            dfs(next, index+1, direction)

        elif direction==0:
            nowx, nowy = now
            next = (nowx, nowy-1)
            dfs(next, index+1, direction)

    elif s[index] == 'T':
        direction1 = (direction + 1)%4
        direction2 = (direction - 1)%4
        dfs(now, index+1, direction1)
        dfs(now, index+1, direction2)


now = (0, 0)
direction = 1
dfs(now, 0, direction)
if (x, y) in finished:
    print('Yes')
else:
    print('No')

# D submit
import sys
sys.setrecursionlimit(9000)
s = input()
x, y = map(int, input().split())

Fl = s.split('T')
cnt = [len(tmp) for tmp in Fl]
xcnt = [tmp for ind, tmp in enumerate(cnt) if ind%2==0]
ycnt = [tmp for ind, tmp in enumerate(cnt) if ind%2==1]

xfinished = []
yfinished = []

def xdfs(index, total):
    if index == len(xcnt)-1:
        total1 = total + xcnt[index]
        total2 = total - xcnt[index]
        xfinished.append(total1)
        xfinished.append(total2)
        return

    else:
        total1 = total + xcnt[index]
        xdfs(index+1, total1)
        total2 = total - xcnt[index]
        xdfs(index+1, total2)
def ydfs(index, total):
    if index == len(ycnt)-1:
        total1 = total + ycnt[index]
        total2 = total - ycnt[index]
        yfinished.append(total1)
        yfinished.append(total2)
        return

    else:
        total1 = total + ycnt[index]
        ydfs(index+1, total1)
        total2 = total - ycnt[index]
        ydfs(index+1, total2)

xdfs(0, 0)
ydfs(0, 0)

if (x in xfinished) and (y in yfinished):
    print('Yes')
else:
    print('No')
############ DPで計算するらしい。。。要復習。


########################################################################################################################################################################
############# ABC083
########################################################################################################################################################################
# A
a, b, c, d = map(int, input().split())
l = a+b
r = c+d
if l<r:
    print('Right')
elif l>r:
    print('Left')
else:
    print('Balanced')

# B
n, a, b = map(int, input().split())
total = 0
li = list(range(1, n+1))
for num in li:
    s = str(li)
    sum_ = 0
    for keta in s:
        sum_ = int(keta)
    if a<=sum_<=b:
        total += num
print(total)    


# C
x, y = map(int, input().split())
total = 1
now = x
while now*2 <= y:
    now *= 2
    total += 1
print(total)
    

# D
s = input()
length = len(s)
k = length
mid = length//2
max_k = None

while k>mid:
    overlap = s[length-k:k]
    pure = len(set(overlap)) == 1
    if pure:
        max_k = k
        print(max_k)
        break
    k -= 1

if not max_k:
    print(mid)

# D 3424320
S = input()
N = len(S)
center_i = N//2
center = S[center_i]

ans = N//2
l = r = center_i
if N%2==0:
    l -= 1
while l >= 0:
    if not (S[l] == center == S[r]):
        break
    l -= 1
    r += 1
    ans += 1
print(ans)

########################################################################################################################################################################
############# ABC083
########################################################################################################################################################################
# A
m = int(input())
print(48-m)


# B
a, b = map(int, input().split())
s = input()
n = len(s)
nums = list(range(10))
point = 0
flag = 'Yes'

if n != a+b+1:
    flag = 'No'
    print(flag)
    exit()

while point<=n-1:
    if point == a:
        if s[point] != '-':
            flag = 'No'
    else:
        if int(s[point]) not in nums:
            flag = 'No'
    point += 1
print(flag)

a,b=map(int,input().split())
s=input()
print("Yes" if s[a]=="-" and s.count("-")==1 else "No")

a, b = map(int, input().split())
s = input()

if (s[a]=='-' and s.count('-')==1):
    print('Yes')
else:
    print('No')
        
        

if s[a] != '-':
    print('No')
    exit()
else:
    split = s.split('-')
    mae, ushiro = split[0], split[1]
    for m in mae:
        if int(m) not in nums:
            print('No')
            exit()
    for u in ushiro:
        if int(u) not in nums:
            print('No')
            exit()
print('Yes')

# C
n = int(input())
cond = []
for _ in range(n-1):
    c = list(map(int, input().split()))
    cond.append(c)

for i in range(n):
    if i == n-1:
        print(0)
    else:
        ela = 0

        for j in range(i, n-1):
            if j == i:
                # noru
                ela += cond[j][1]
            elif j != i:
                # noru
                if ela <= cond[j][1]:
                    ela = cond[j][1]
                elif ela > cond[j][1]:
                    k = cond[j][2]
                    if ela%k == 0:
                        pass
                    else:
                        ela = ela + (k-(ela%k))               
            # susumu
            ela += cond[j][0]
        print(ela)

# D
import numpy as np

def seachPrimeNum(N):
    max = int(np.sqrt(N))
    seachList = [i for i in range(2,N+1)]
    primeNum = []
    while seachList[0] <= max:
        primeNum.append(seachList[0])
        tmp = seachList[0]
        seachList = [i for i in seachList if i % tmp != 0]
    primeNum.extend(seachList)
    return primeNum

eratos = seachPrimeNum(100000)
ruiseki = [0]
for i in range(2, 100001):
    num = ruiseki[-1]
    if [i, i//2] in eratos:
        num += 1
    ruiseki.append(num)
print(ruiseki)



print(eratos)

q = int(input())
for _ in range(q):
    l, r = map(int, input().split())
    cand = []
    for m in range(l, r+1):
        if isPrime(m):
            cand.append((m+1)//2)
    total = 0
    for c in cand:
        if isPrime(c):
            total += 1
    print(total)

# D
from math import sqrt
def sieve(n:int)->list:
    """エラトステネスの篩"""
    if n<2:  return [False]*(n+1)
    is_prime = [True]*(n+1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(sqrt(n))+1):
        if is_prime[i]:
            for j in range(i*2, n+1, i):
                is_prime[j] = False
    
    return is_prime


N = int(1e5)
is_prime = sieve(N)
# like-2017
cnt = [0]*(N+1)
for i in range(1, N+1):
    cnt[i] += cnt[i-1]
    if is_prime[i] and is_prime[(i+1)//2]:
        cnt[i] += 1
    
import sys
Q = sys.stdin.readline()
for l,r in (map(int, line.split()) for line in sys.stdin.readlines()):
    print(cnt[r] - cnt[l-1])

# D submit
from math import sqrt
def sieve(n):
    if n<2:
        return [False]*(n+1)
    is_prime = [True]*(n+1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(sqrt(n))+1):
        if is_prime[i]:
            for j in range(i*2, n+1, i):
                is_prime[j] = False
    return is_prime

n = int(1e5)
is_prime = sieve(n)

cnt = [0]*(n+1)
for i in range(1, N+1):
    cnt[i] += cnt[i-1]
    if is_prime[i] and is_prime[(i+1)//2]:
        cnt[i] += 1
q = int(input())
for _ in range(q):
    l, r = map(int, input().split())
    print(cnt[r] - cnt[l-1])

########################################################################################################################################################################
############# ABC085
########################################################################################################################################################################
# A
s = input()
s = s.replace('2017', '2018')
print(s)

# B
n = int(input())
mochis = []
for _ in range(n):
    mochis.append(int(input()))
print(len(set(mochis)))



# C
n, y = map(int, input().split())
k = y//1000
candidate = []

for i in range(n+1):
    for j in range(n+1-i):
        if 9*i + 4*j == k-n:
            candidate.append((i, j))

for cand in candidate:
    i, j = cand[0], cand[1]
    if n-i-j>=0:
        t = n-i-j
        print(i, j, t)
        exit()

print('-1 -1 -1')


# D

########################################################################################################################################################################
############# ABC086
########################################################################################################################################################################
# A
a, b = map(int, input().split())
if a*b%2 == 0:
    print('Even')
else:
    print('Odd')

# B
import math
a, b = map(int, input().split())
t = int(str(a) + str(b))
troot = math.sqrt(t)
if troot.is_integer():
    print('Yes')
else:
    print('No')

# C
n = int(input())
plan = []
for _ in range(n):
    t, x, y = map(int, input().split())
    plan.append((t, x, y))

flag = 'Yes'
elapsed = 0
now = (0, 0)
for p in plan:
    t, x, y = p[0], p[1], p[2]
    nx, ny = now[0], now[1]

    time = t - elapsed
    distance = abs(x - nx) + abs(y - ny)
    buff = time - distance
    if (buff>=0) and (buff%2 == 0):
        elapsed = t
        now = (x, y)
    else:
        flag = 'No'
        break
print(flag)

# D
import numpy as np
n, k = map(int, input().split())
order = []
for _ in range(n):
    x, y, c = map(int, input().split())
    if c == 'W':
        y = y+k
        c = 'B'

    order.append((x%2*k, y%2*k, c))

for i in range(2*k):
    for j in range(2*k):

# D reference
import numpy as np
N, K = map(int, input().split())
mod = 2*K
field = [[0]*(2*K+1) for _ in range(2*K+1)]

def gen_pattern(x):
    if x < K-1:
        return [[0, x+1], [x+K+1, 2*K]]
    else:
        return [[x-K+1, x+1]]

for _ in range(N):
    x, y, c = input().split()
    x, y = int(x), int(y)
    if c == 'W':
        y += K
    x %= mod
    y %= mod
    for tmp in [0, K]:
        for l, r in gen_pattern((x+tmp) % mod):
            for b, t in gen_pattern((y+tmp) % mod):
                field[l][b] += 1
                field[l][t] -= 1
                field[r][b] -= 1
                field[r][t] += 1
print(np.max(np.cumsum(np.cumsum(field, axis=0), axis=1)))

########################################################################################################################################################################
############# ABC086
########################################################################################################################################################################
# A
a, b = map(int, input().split())
la = len(str(a))
lb = len(str(b))
if la==1 and lb==1:
    print(a*b)
else:
    print('-1')

# B
n = int(input())
flag = False

for i in range(1, n+1):
    if n%i==0:
        counter = n//i
        if len(str(i))==1 and len(str(counter))==1:
            flag = True
if flag:
    print('Yes')
else:
    print('No')

# C
def factorization(n):
    arr = []
    temp = n
    for i in range(2, int(-(-n**0.5//1))+1):
        if temp%i==0:
            cnt=0
            while temp%i==0:
                cnt+=1
                temp //= i
            arr.append([i, cnt])

    if temp!=1:
        arr.append([temp, 1])

    if arr==[]:
        arr.append([n, 1])

    return arr

n = int(input())

import math
n = int(input())
sn = int(math.sqrt(n)) + 1

seki = []
for i in range(1, sn):
    if n%i == 0:
        seki.append([i, n//i])
totalseki = [sum(li) for li in seki]
min_ = min(totalseki)

print(min_ - 2)

# D
import math
a, b, x = map(int, input().split())

tan1 = (2/a)*(b - (x/(a**2)))
tan2 = (1/2)*(a*(b**2)/x)

actan = math.atan(tan1)
moto = (x/a**2)
gen = (a*math.tan(actan))
judge = (moto-gen) >= 0

if judge:
    tan = tan1
else:
    tan = tan2

actan = math.atan(tan)
degree = math.degrees(actan)
print(degree)

# E

# F

########################################################################################################################################################################
############# ABC089
########################################################################################################################################################################
# A
n = int(input())
print(n//3)

# B
n = int(input())
sli = list(input().split())
for s in sli:
    if s == 'Y':
        print('Four')
        exit()

print('Three')

# C
from itertools import combinations
n = int(input())
names = [0 for _ in range(5)]
for _ in range(n):
    start = input()[0]
    if start == 'M':
        names[0] += 1
    if start == 'A':
        names[1] += 1
    if start == 'R':
        names[2] += 1
    if start == 'C':
        names[3] += 1
    if start == 'H':
        names[4] += 1
comb = combinations(names, 3)
total = 0
for c in comb:
    total += c[0]*c[1]*c[2]
print(total)

# D
import numpy as np
def dijkstra(start, node_num, cost):
    '''
    start:始点
    node_num:ノード数
    cost: cost[u][v]がuからvへの重みとなるような配列
    -------------------------------------------
    dist: i番目が始点から頂点iまでの最短距離になる配列
    '''
    dist = [float('inf')] * node_num
    used = [False] * node_num
    dist[start] = 0

    while True:
        v = -1
        #まだ使われてない頂点の中から最小の距離のものを探す
        for i in range(node_num):
            if (not used[i]) and (v == -1):
                v = i
            elif (not used[i]) and dist[i] < dist[v]:
                v = i

        ### 全ての頂点が使われてた場合は終了
        if v == -1:
            break
        ### 探してきた頂点を使って距離の更新
        used[v] = True
        for i in range(node_num):
            dist[i] = min(dist[i], dist[v] + cost[v][i])
        
    return dist

h, w, d = map(int, input().split())
field = []
for _ in range(h):
    wi = list(map(int, input().split()))
    field.append(wi)
field = np.array(field)

cost = [[float('inf') for _ in range(h*w)] for _ in range(h*w)]
for hi in range(h):
    for wi in range(w):
        nodenum = field[hi, wi] - 1
        for dh, dw in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            if (0 <= hi+dh <= h-1) and (0<= wi+dw <= w-1):
                cost[nodenum][field[hi+dh, wi+dw]-1] = 1
dijk = []
for i in range(h*w):
    dijk.append(dijkstra(start=i, node_num=h*w, cost=cost))

def search(dijk, l, r, d):
    mana = 0
    while l < r:
        mana += dijk[l-1][l+d-1]
        l += d
    print(mana)

question = int(input())
for _ in range(question):
    l, r = map(int, input().split())
    search(dijk=dijk, l=l, r=r, d=d)

########################################################################################################################################################################
############# ABC089
########################################################################################################################################################################
# A
n = int(input())
if n % 2 == 0:
    print(n//2-1)
else:
    print(n//2)

# B
from collections import Counter
mod = 998244353
n = int(input())
dli = list(map(int, input().split()))
for ind in range(len(dli)):
    if (ind == 0):
        if (dli[ind] != 0):
            print(0)
            exit()
    else:
        if (dli[ind] == 0):
            print(0)
            exit()

count = Counter(dli)
# 木である条件
for ind, key in enumerate(count.keys()):
    if ind != key:
        print(0)
        exit()

sort = sorted(count.items(), key=lambda x:x[0])

tmp = []
for ind in range(len(sort)-1):
    mae = sort[ind][1]
    next = sort[ind+1][1]

    tmp.append(int(mae**next))

total = 1
for t in tmp:
    total *= t
print(total%mod)

# C
n = int(input())
al = list(map(int, input().split()))
bl = list(map(int, input().split()))

sal = sorted(al)
sbl = sorted(bl)
for sa, sb in zip(sal, sbl):
    if sa > sb:
        print('No')
        exit()

cl = [a - b for a, b in zip(al, bl)]
plustotal = 0
for c in cl:
    if c > 0:
        plustotal += 1
if plustotal >= n-1:
    print('No')
    exit()

print('Yes')

# D

# E

# F

########################################################
##### 暇つぶし
#############################################################
n = int(input())
h = int(input())
w = int(input())

hbaf = n - h
wbaf = n - w
print(nbaf*wbaf)

n = int(input())
a, b = map(int, input().split())
pli = list(map(int, input().split()))

one = [p for p in pli if p<=a]
two = [p for p in pli if (a+1)<=p<=b]
the = [p for p in pli if (b+1)<=p]

print(min([len(one), len(two), len(the)])

##########################################################################################
########## ABC100
##########################################################################################
# A
a, b = map(int, input().split())
if max(a, b) > 8:
    print(':(')
else:
    print('Yay!')

# B
d, n = map(int, input().split())
moto = 100**d
if n==100:
    n+= 1
print(moto*n)

# C
n = int(input())
ali = list(map(int, input().split()))
total = 0
for a in ali:
    while a%2==0:
        a = (a//2)
        total += 1
print(total)

# C
n, m = map(int, input().split())
lst = [[] for _ in range(8)]
for _ in range(n):
    index = 0
    x, y, z = map(int, input().split())
    for nx in [x, -x]:
        for ny in [y, -y]:
            for nz in [z, -z]:
                lst[index].append(nx+ny+nz)
                index += 1
sortlst = []
for l in lst:
    sortlst.append(sorted(l, reverse=True))
msums = []
for sl in sortlst:
    msums.append(sum(sl[:m]))
print(max(msums))

##########################################################################################
########## ABC147
###############################################################################


# A
a, b, c = map(int, input().split())
su = a + b + c
if su >= 22:
    print('bust')
else:
    print('win')

# B
s = input()
leng = len(s)
if leng%2==0:
    mae = s[:leng//2]
    ushi = s[leng//2:][::-1]
else:
    mae = s[:leng//2]
    ushi = s[leng//2+1:][::-1]

hagu = 0
for m, u in zip(mae, ushi):
    if m != u:
        hagu += 1
print(hagu)
    

# C
from collections import deque
n = int(input())
lst = [[] for _ in range(n)]
for j in range(n):
    nums = int(input())
    for _ in range(nums):
        x, y = map(int, input().split())
        x -= 1
        lst[j].append([x, y])

totals = []
for i in range(n):
    log = [float('inf') for _ in range(n)]
    log[i] = 1
    
    d = deque()
    for k in lst[i]:
        d.append(k)
    
    conf = False
    while d:
        man, judge = d.popleft()
        if log[man] == float('inf'):
            log[man] = judge
        else:
            if log[man] != judge:
                conf = True
                break
        for k in lst[man]:
            d.append(k)
    
    if conf:
        totals.append(0)
    else:
        mlog = []
        for l in log:
            if l == float('inf'):
                mlog.append(1)
            else:
                mlog.append(l)
        totals.append(sum(mlog))
print(max(totals))

# C
from collections import deque
n = int(input())
lst = [[] for _ in range(n)]
for j in range(n):
    nums = int(input())
    for _ in range(nums):
        x, y = map(int, input().split())
        x -= 1
        lst[j].append([x, y])

totals = []
for i in range(n):
    log = [float('inf') for _ in range(n)]
    log[i] = 1

# D
import numpy as np
n = int(input())
ali = list(map(int, input().split()))
mod = 10**9 + 7

zen = 0
for i in range(n):
    zen ^= ali[i]

total = zen
for i in range(n):
    total += zen^ali[i]
print(total%mod)

# E
ali = [2,3,4]
total = 0
for a in ali:
    total ^= a
print(total)
totawi = total ^ 2
print(totawi)


### AGC
s = input()
partial = []
n = len(s)
point = 0
val = 0
dirction = s[0]
a = 0
while point < n:
    while s[point] == direction:
        if s[point] == '<':
            a += 1
        else:
            a -= 1


s = input()
n = len(s)
partial = []
point = 0
direction = s[0]
while point < n:
    temp = point
    while True:
        if point == n:
            partial.append(s[temp:point+1])
        if s[point+1] != direction:
            partial.append(s[temp:point+1])
            break
        point += 1
        

########## ABC 103
# A
lst = list(map(int, input().split()))
sor = sorted(lst)
cost = 0
for i in range(len(sor)):
    if i == 0:
        continue
    else:
        cost += abs(sor[i]-sor[i-1])
print(cost)

# B
s = input()
t = input()
n = len(s)
flag = 'No'
for _ in range(n):
    s = s[-1] + s[:-1]
    if s == t:
        flag = 'Yes'
        break
print(flag)

# C
n = int(input())
lst = list(map(int, input().split()))

total = 0
for i in lst:
    total += (i-1)
print(total)

# D
n, m = map(int, input().split())
lst = []
for _ in range(m):
    a, b = map(int, input().split())
    lst.append((a, b))
sor = sorted(lst, key=lambda x:x[1])
spl = -1
cnt = 0
for a, b in sor:
    if a<=spl:
        continue
    else:
        spl = b-1
        cnt += 1
print(cnt)


