def viterbi(x, w, y = []):
    w = list(w)
    lenx = len(x)
    x = x.reshape((lenx, FBANKS))
    observation = np.array(w[:PHONES*FBANKS]).reshape((PHONES, FBANKS))
    trans = np.array(w[PHONES*FBANKS:]).reshape((PHONES, PHONES))
    xobs = np.dot(x, observation.T)
    prob_pre = np.zeros((PHONES, 1))
    trace = []
    for i in range(lenx):
        prob_now = prob_pre + trans + xobs[i, :]
        if len(y) > 0:
            prob_now[:,y[i]] -= 1
        argmax = np.argmax(prob_now, axis = 0)
        prob_pre = np.max(prob_now, axis = 0).reshape((PHONES, 1))
        trace.append(argmax)
    now = np.argmax(prob_pre)
    ans = []
    ans.append(now)
    for i in range(lenx-1, 0, -1):
        now = trace[i][now]
        ans.append(now)
    return np.array(ans[::-1])
