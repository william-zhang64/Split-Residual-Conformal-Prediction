"""Adaptive conformal prediction helpers used by the synthetic, stocks, and
manheim adaptive notebooks. Extracted from the original
`conformal_exp_heuristic_..._dtaci.ipynb` cells 1-4.
"""
import numpy as np
from scipy.stats import binom
from statsmodels.tsa.forecasting.theta import ThetaModel
from pathlib import Path
from tqdm import tqdm

# rpy2 is only required if you call `dtaci`. We import lazily so users who
# only want the Python methods don't need R installed. `_load_dtaci()`
# caches the loaded R modules on first call.
_DTACI_STATE = {"loaded": False}


def _load_dtaci(r_dir="DtACI-main"):
    """Lazily load AgACI.R + DtACI.R via rpy2 and cache the resulting names."""
    if _DTACI_STATE["loaded"]:
        return _DTACI_STATE
    import rpy2.robjects as ro
    from rpy2.robjects import FloatVector, numpy2ri
    from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
    from rpy2.robjects.conversion import localconverter
    with open(Path(r_dir) / "AgACI.R") as f:
        agACI_src = f.read()
    with open(Path(r_dir) / "DtACI.R") as f:
        dtACI_src = f.read()
    _DTACI_STATE.update(
        loaded=True,
        dtACI_funcs=SignatureTranslatedAnonymousPackage(dtACI_src, "dtACI"),
        agACI_funcs=SignatureTranslatedAnonymousPackage(agACI_src, "agACI"),
        FloatVector=FloatVector,
        ro=ro,
        localconverter=localconverter,
        numpy2ri=numpy2ri,
    )
    return _DTACI_STATE

def sliding_window_average(coverage, window_size, mode='valid'):
    window = np.ones(window_size) / window_size
    return np.convolve(coverage, window, mode=mode)

def fixed_sequence_testing(lambda_set, pvals, delta, initializations, pvals_other = []):
    lambda_val = []
    for j in initializations:
        lambda_j = tuple(lambda_set[j,:])
        if pvals[j]<= delta/len(initializations):
            if(pvals_other):
                if pvals_other[j]<= delta/len(initializations):
                    lambda_val.append(lambda_j)
            else:

                lambda_val.append(lambda_j)

    return lambda_val

def filter_larger_tuples_np(tuples, ref_tuple, index):
    mask = tuples[:, index] > ref_tuple[index]
    if np.sum(mask) == 0:
        return ref_tuple.reshape(1,-1)

    return tuples[mask]

def filter_smaller_tuples_np(tuples, ref_tuple, index):
    mask = tuples[:, index] < ref_tuple[index]
    if np.sum(mask) == 0:
        return ref_tuple.reshape(1,-1)

    return tuples[mask]

def find_right_tuple_no_cov(lambda_val, prevab,covered, covereds_a, covereds_b):
    # selects tuple of a,b heuristically among valid set
    tuples_array = np.array(lambda_val)
    target_array = np.array(prevab)
    epsilon_selector = [0,0]
    if covereds_b >= covereds_a:
        first_dim = 1
        second_dim = 0
        first_covereds = covereds_b
        second_covereds = covereds_a
    else:
        first_dim = 0
        second_dim = 1
        first_covereds = covereds_a
        second_covereds = covereds_b

    if first_covereds > 0 or not covered:
        tuples_consider = filter_larger_tuples_np(tuples_array, target_array, first_dim)
        if len(tuples_consider) == 0:
            epsilon_selector[first_dim] = -1
            tuples_consider = tuples_array[tuples_array[:, first_dim] == np.max(tuples_array[:,first_dim])]

    else:
        tuples_consider = filter_smaller_tuples_np(tuples_array, target_array, first_dim)
        if len(tuples_consider) == 0:
            epsilon_selector[first_dim] = 1

            tuples_consider = tuples_array[tuples_array[:, first_dim] == np.min(tuples_array[:,first_dim])]

    if second_covereds > 0 or not covered:
        tuples_consider_second = filter_larger_tuples_np(tuples_consider, target_array, second_dim)
        if len(tuples_consider_second) == 0:
            epsilon_selector[second_dim] = -1
            return tuples_consider[np.argmax(tuples_consider)[:,first_dim]], epsilon_selector
        else:
            if first_covereds == 0:
                min_first = np.min(tuples_consider_second[:, first_dim])
                min_first_candidates = tuples_consider_second[tuples_consider_second[:, first_dim] == min_first]
                min_second = np.min(min_first_candidates[:, second_dim])
                final_candidate = min_first_candidates[min_first_candidates[:, second_dim] == min_second][0]
                return final_candidate, epsilon_selector
            else:

                min_first = np.min(tuples_consider_second[:, first_dim])
                min_first_candidates = tuples_consider_second[tuples_consider_second[:, first_dim] == min_first]
                max_second = np.max(min_first_candidates[:, second_dim])
                final_candidate = min_first_candidates[min_first_candidates[:, second_dim] == max_second][0]
                return final_candidate, epsilon_selector
    else:
        tuples_consider_second = filter_smaller_tuples_np(tuples_consider, target_array, second_dim)
        if len(tuples_consider_second) == 0:
            epsilon_selector[second_dim] = 1
            return tuples_consider[np.argmin(tuples_consider)[:,second_dim]], epsilon_selector
        else:
            if first_covereds > 0:
                min_first = np.min(tuples_consider_second[:, first_dim])
                min_first_candidates = tuples_consider_second[tuples_consider_second[:, first_dim] == min_first]
                max_second = np.max(min_first_candidates[:, second_dim])
                final_candidate = min_first_candidates[min_first_candidates[:, second_dim] == max_second][0]
                return final_candidate, epsilon_selector
            else:
                max_first = np.max(tuples_consider_second[:, first_dim])
                max_first_candidates = tuples_consider_second[tuples_consider_second[:, first_dim] == max_first]
                max_second = np.max(max_first_candidates[:, second_dim])
                final_candidate = max_first_candidates[max_first_candidates[:, second_dim] == max_second][0]
                return final_candidate, epsilon_selector

def adaptive_fwer_alpha(
    #our split residual method that performs fwer to obtain a and b weights for residual components
    scores,
    alpha,
    epsilon,
    window_length,
    cal_size,
    T_burnin,
    lambda_set,
    r1_scores,
    r2_scores,
    delta,
    initializations,
    ahead,
    lr,
    alpha_lr,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    fake_covereds = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    covereds_r2 = np.zeros((T_test,))
    covereds_r1 = np.zeros((T_test,))
    preva = np.ones((T_test,))
    prevb = np.ones((T_test,))

    r1_store = np.zeros((T_test,))
    r2_store = np.zeros((T_test,))
    epsilon_1t = epsilon
    epsilon_2t = epsilon
    alphat = alpha
    for t in tqdm(range(T_test)):
        t_pred = t- ahead + 1
        prev_window = scores[max(t_pred - window_length, 0):t_pred]
        prev_window_r1 = r1_scores[max(t_pred - window_length, 0):t_pred]
        prev_window_r2 = r2_scores[max(t_pred - window_length, 0):t_pred]
        if t_pred > T_burnin:
            cal_set = prev_window
            conf_set_r1 = prev_window_r1
            conf_set_r2 = prev_window_r2

            delta_r1 = np.quantile(conf_set_r1, 1 - epsilon_1t)
            r2 = np.quantile(conf_set_r2, 1 - epsilon_2t)
            r1_store[t] = delta_r1
            r2_store[t] = r2
            pvals = []
            for a,b in lambda_set:
                width = delta_r1*a + r2*b
                empirical_cov = np.sum(cal_set > width)
                pval = binom.cdf(empirical_cov, cal_size, np.clip(alphat,0,1))
                pvals.append(pval)
            if initializations is None:
                initializations = np.arange(len(lambda_set))
            lambda_val = fixed_sequence_testing(lambda_set, pvals, delta, initializations)

            if(len(lambda_val)==0):
                if(alphat > 1):
                    qs[t] = -np.inf
                    epsilon_1t = min(1, epsilon_1t+lr)
                    epsilon_2t = min(1, epsilon_2t+lr)
                    a_sel = preva[t_pred-1]
                    b_sel = prevb[t_pred-1]
                    covereds[t] = 0
                    fake_covereds[t] = 0
                    covereds_r1[t] = 0 if (a_sel*delta_r1>= r1_scores[t]) else r1_scores[t] - a_sel*delta_r1
                    covereds_r2[t] = 0 if (b_sel*r2>= r2_scores[t]) else r2_scores[t] - b_sel*r2
                    preva[t] = a_sel
                    prevb[t] = b_sel
                    alphat = alphat - alpha_lr*alpha
                else:

                    qs[t] = -np.inf
                    epsilon_1t = max(0.001, epsilon_1t-lr)
                    epsilon_2t = max(0.001, epsilon_2t-lr)
                    a_sel = preva[t_pred-1]
                    b_sel = prevb[t_pred-1]
                    covereds[t] = 0
                    fake_covereds[t] = 1
                    covereds_r1[t] = 0 if (a_sel*delta_r1>= r1_scores[t]) else r1_scores[t] - a_sel*delta_r1
                    covereds_r2[t] = 0 if (b_sel*r2>= r2_scores[t]) else r2_scores[t] - b_sel*r2
                    preva[t] = a_sel
                    prevb[t] = b_sel
                    alphat = alphat + alpha_lr*alpha
            else:
                tup, epsilon_select = find_right_tuple_no_cov(lambda_val, (preva[t_pred-1], prevb[t_pred-1]),covereds[t_pred-1], covereds_r1[t_pred-1], covereds_r2[t_pred-1])
                a_sel,b_sel = tup.ravel()
                qs[t] = a_sel*delta_r1 + b_sel*r2

                covereds[t] = qs[t]>= scores[t]
                fake_covereds[t] = covereds[t]
                covereds_r1[t] = 0 if (a_sel*delta_r1>= r1_scores[t]) else r1_scores[t] - a_sel*delta_r1
                covereds_r2[t] = 0 if (b_sel*r2>= r2_scores[t]) else r2_scores[t] - b_sel*r2
                preva[t] = a_sel
                prevb[t] = b_sel

                if alphat>1:
                    epsilon_1t = min(1, epsilon_1t+lr)
                    epsilon_2t = min(1, epsilon_2t+lr)
                    alphat = alphat - alpha_lr*(1 -alpha)
                else:
                    if(epsilon_select[0]>0):
                        epsilon_1t = min(1, epsilon_1t+lr)
                    elif(epsilon_select[0]<0):
                        epsilon_1t = max(0.001, epsilon_1t-lr)
                    if(epsilon_select[1]>0):
                        epsilon_2t = min(1, epsilon_2t+lr)
                    elif(epsilon_select[1]<0):
                        epsilon_2t = max(0.001, epsilon_2t-lr)
                    if(fake_covereds[t_pred-1]==0):
                        alphat = alphat - alpha_lr*(1 -alpha)
                    else:
                        alphat =  alphat + alpha_lr*(alpha)

        else:
            if t_pred < np.ceil(1/alpha):
                qs[t] = -np.inf
                preva[t] = 1
                prevb[t] = 1
                qs[t] = -np.inf
                covereds[t] = 0
                fake_covereds[t] = covereds[t]
                covereds_r1[t] = 1
                covereds_r2[t] = 1
                r1_store[t] = -np.inf
                r2_store[t] = -np.inf
            else:
                preva[t] = 1
                prevb[t] = 1
                delta_r1 =  np.quantile(r1_scores,1-alpha)
                r2 = np.quantile(r2_scores,1-alpha)
                qs[t] = delta_r1 + r2
                covereds[t] = 0
                fake_covereds[t] = covereds[t]
                covereds_r1[t] = 0 if (delta_r1>= r1_scores[t]) else r1_scores[t] - delta_r1
                covereds_r2[t] = 0 if (r2>= r2_scores[t]) else r2_scores[t] - r2
                r1_store[t] = delta_r1
                r2_store[t] = r2
    results = { "method": "Two-stage", "q" : qs, "a" : preva, "b" : prevb, "cov":covereds, "r1":r1_store, "r2":r2_store, "cov_r1":covereds_r1, "cov_r2":covereds_r2}
    return results

def dtaci(scores, alpha, gammas, ahead=1, window_size=100, r_dir="DtACI-main"):
    # implements dtaci baseline
    state = _load_dtaci(r_dir)
    FloatVector = state["FloatVector"]
    ro = state["ro"]
    localconverter = state["localconverter"]
    numpy2ri = state["numpy2ri"]
    dtACI_funcs = state["dtACI_funcs"]

    T = scores.shape[0]

    gammas_r = FloatVector(gammas)
    alpha_r = ro.FloatVector([alpha])[0]
    ahead_r = ro.FloatVector([ahead])[0]
    scores_r = FloatVector(scores.tolist())

    result = dtACI_funcs.conformalAdaptStable(
        betas=scores_r,
        alpha=alpha_r,
        gammas=gammas_r,
        ahead=ahead_r
    )

    with localconverter(numpy2ri.converter):
        alphaSeq = np.array(result[0])
        errSeqAdapt = np.array(result[1])
        errSeqFixed = np.array(result[2])
        gammaSeq = np.array(result[3])
        meanAlphaSeq = np.array(result[4])
        meanErrSeq = np.array(result[5])
        meanGammas = np.array(result[6])

    qs = np.zeros(T)
    covereds = np.zeros(T)

    for t in range(T):
        t_pred = t - ahead + 1
        if t_pred >= 1:

            quantile_level = np.clip(alphaSeq[t_pred], 0, 1)
            qs[t] = np.quantile(scores[:t_pred], quantile_level, method="higher")
            covereds[t] = qs[t] >= scores[t]
        else:
            qs[t] = np.inf

    results = {
        "Method": "DtACI",
        "q": qs,
        "cov": covereds,
        "alpha": alphaSeq,
        "gamma": gammaSeq,
    }

    return results

def aci_clipped(
    # implements aci baseline
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        clip_value = scores[max(t_pred-window_length,0):t_pred].max() if t_pred > 0 else np.inf
        if t_pred > T_burnin:
            if alphat <= 1/(t_pred+1):
                qs[t] = np.inf
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.inf
        if qs[t] == np.inf:
            qs[t] = clip_value
    results = { "method": "ACI (clipped)", "q" : qs, "alpha" : alphas, "cov":covereds}
    return results

def aci(
    # implements aci baseline (not clipped)
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            if alphat <= 1/(t_pred+1):
                qs[t] = np.inf
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.inf
    results = { "method": "ACI", "q" : qs, "alpha" : alphas, "cov":covereds}
    return results

def mytan(x):
    if x >= np.pi/2:
        return np.inf
    elif x <= -np.pi/2:
        return -np.inf
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI):
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)))
    out = KI * tan_out
    return  out

def quantile_integrator_log_scorecaster(
    # implements conformal PID
    scores,
    alpha,
    lr,
    data,
    T_burnin,
    Csat,
    KI,
    upper,
    ahead,
    integrate=True,
    proportional_lr=True,
    scorecast=True,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    scorecasts = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    seasonal_period = kwargs.get('seasonal_period')
    if seasonal_period is None:
        seasonal_period = 1
    try:
        if 'scorecasts' in data.columns:
            scorecasts = np.array([s[int(upper)] for s in data['scorecasts'] ])
            train_model = False
        else:
            scorecasts = np.load('./.cache/scorecaster/' + kwargs.get('config_name') + '_' + str(upper) + '.npy')
            train_model = False
    except:
        train_model = True
    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue
        covereds[t] = qs[t] >= scores[t]
        grad = alpha if covereds[t_pred] else -(1-alpha)
        integrator_arg = (1-covereds)[:t_pred].sum() - (t_pred)*alpha
        integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI)
        if scorecast and train_model and t_pred > T_burnin and t+ahead < T_test:
            curr_scores = np.nan_to_num(scores[:t_pred])
            model = ThetaModel(
                    curr_scores.astype(float),
                    period=seasonal_period,
                    ).fit()
            scorecasts[t+ahead] = model.forecast(ahead).iloc[-1]
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = integrator if integrate else 0
            qs[t+1] = qts[t+1] + integrators[t+1]
            if scorecast:
                qs[t+1] += scorecasts[t+1]
    results = {"method": "Quantile+Integrator (log)+Scorecaster", "q" : qs, "cov": covereds}
    return results

def get_online_quantile(scores, q_1, etas, alpha, ahead):
    T = scores.shape[0]
    q = np.zeros(T)
    q[0] = q_1
    for t in range(T):
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue

        err_t = (scores[t_pred-1] > q[t_pred-1]).astype(int)
        q[t] = q[t_pred-1] - etas[t] * (alpha - err_t)
    return q

def decay_quantile(scores, q_1,etas,alpha, ahead):
    # implements OCID baseline
    qs = get_online_quantile(scores, q_1, etas, alpha,ahead)
    covereds = qs >= scores.flatten()
    results = {"Method":"Decaying Weights", "q":qs, "cov":covereds}
    return results

def ECI(
    # implements ECI baseline
    scores,
    alpha,
    lr,
    T_burnin,
    ahead,
    proportional_lr=True,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    c = 1

    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue

        covereds[t] = qs[t] >= scores[t]

        grad = alpha if covereds[t_pred] else -(1-alpha)

        eq = (scores[t_pred]-qs[t_pred])*c*np.exp(-c*(scores[t_pred]-qs[t_pred]))/((1+np.exp(-c*(scores[t_pred]-qs[t_pred]))) ** 2)
        integrator = np.mean(eq)

        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = lr_t * integrator
            qs[t+1] = qts[t+1] + integrators[t+1]

    results = {"method": "ECI", "q" : qs,'cov':covereds}

    return results
