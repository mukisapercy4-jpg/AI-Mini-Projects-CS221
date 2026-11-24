# Assignment 5: Pac-Man, written answers (HW5)

**Authors:** Mukisa Percy & Munanura Jill

Setup: there are n + 1 agents a_0, ..., a_n, where a_0 is Pac-Man (MAX) and a_1, ..., a_n are ghosts. One depth d consists of Pac-Man and every ghost each moving once, so the depth drops only after the last ghost a_n has moved. Notation: IsEnd(s), Utility(s), Eval(s), Player(s), Actions(s), Succ(s, a).

---

## Problem 1a: Minimax recurrence

$$
V_{\text{minmax}}(s, d) =
\begin{cases}
\text{Utility}(s) & \text{if } \text{IsEnd}(s) \\
\text{Eval}(s) & \text{if } d = 0 \\
\max_{a \in \text{Actions}(s)} V_{\text{minmax}}(\text{Succ}(s, a), d) & \text{if } \text{Player}(s) = a_0 \\
\min_{a \in \text{Actions}(s)} V_{\text{minmax}}(\text{Succ}(s, a), d) & \text{if } \text{Player}(s) = a_i,\ 1 \le i \le n - 1 \\
\min_{a \in \text{Actions}(s)} V_{\text{minmax}}(\text{Succ}(s, a), d - 1) & \text{if } \text{Player}(s) = a_n
\end{cases}
$$

The end-state check comes first, so a finished game returns its final score even when d = 0. Only the last ghost a_n lowers d, because a full depth means every agent has moved once.

---

## Problem 3a: Expectimax recurrence

Each ghost chooses uniformly at random from its legal moves, so every ghost layer becomes an average:

$$
V_{\text{exptmax}}(s, d) =
\begin{cases}
\text{Utility}(s) & \text{if } \text{IsEnd}(s) \\
\text{Eval}(s) & \text{if } d = 0 \\
\max_{a \in \text{Actions}(s)} V_{\text{exptmax}}(\text{Succ}(s, a), d) & \text{if } \text{Player}(s) = a_0 \\
\dfrac{1}{|\text{Actions}(s)|} \displaystyle\sum_{a \in \text{Actions}(s)} V_{\text{exptmax}}(\text{Succ}(s, a), d) & \text{if } \text{Player}(s) = a_i,\ 1 \le i \le n - 1 \\
\dfrac{1}{|\text{Actions}(s)|} \displaystyle\sum_{a \in \text{Actions}(s)} V_{\text{exptmax}}(\text{Succ}(s, a), d - 1) & \text{if } \text{Player}(s) = a_n
\end{cases}
$$

---

## Problem 5: AI (mis)alignment and reward hacking

**(a) Why do minimax and expectimax behave differently on `trappedClassic`? (2 points)**

Minimax assumes every ghost plays optimally, always choosing the move that is worst for Pac-Man, so in the trapped layout it believes it will be caught whatever it does. Since every step also costs a point, the best outcome it can find is to die as soon as possible, which is why it rushes the closest ghost. Expectimax assumes the ghosts move randomly, so it sees a real chance that they move away and let Pac-Man escape and eat the remaining food, and the expected value of trying beats a certain loss.

**(b) A change to the evaluation or utility function that stops minimax dying instantly (1 point)**

Add a reward term for food eaten or remaining food to the evaluation function, so that among lines where Pac-Man is doomed it prefers the one that eats the most pellets first. Alternatively, make the penalty for losing a fixed amount that does not depend on how long the game lasted, so that dying sooner is no longer better than dying later.

**(c) Another example of reward hacking or negative side effects (2 points)**

A delivery app that rewards riders for the most deliveries per hour can lead riders to run red lights and ride unsafely to hit the number. This is **reward hacking**: the riders satisfy the stated objective (deliveries per hour) in a way that defeats the designer's real goal of fast and safe delivery.