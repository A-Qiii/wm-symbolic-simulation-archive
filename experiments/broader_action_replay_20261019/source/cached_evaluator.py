def probabilities(m, state, b, samples):
    p = m.symbols(state)
    a = torch.cat([p['objects'].float().softmax(-1).flatten(-2), p['drawer'].float().softmax(-1), p['slider'].float().softmax(-1), p['lights'].float().sigmoid()], -1)
    return a.reshape(b, samples, 32).mean(1).cpu().numpy()

def savearrays(path, rows):
    assert rows
    np.savez_compressed(path, **{k: np.array([r[k] for r in rows]) for k in rows[0]})

def assess(m, data, arm, seed):
    torch.manual_seed(81831)
    records = []
    features = []
    qprobs = []
    hstates = []
    begin = time.perf_counter()
    sample = 8
    with torch.inference_mode():
        for off in range(0, len(data.episodes), 4):
            ids = list(range(off, min(off + 4, len(data.episodes))))
            b = data.batch(ids, 'cuda')
            n = len(ids)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                enc = data.cached_encoded(ids, 'cuda')
                actions = m.action(b['skill'], b['arguments'], b['budget'])
                dp, dq, _, _ = m.history(enc[:, :5], actions[:, :4], True)
                hs = torch.stack([s.h for s in dp], 1).float().cpu().numpy()
                qq = torch.stack(dq, 1).float().cpu().numpy()
                ee = enc[:, :5].float().cpu().numpy()
                posts, _, _, _ = m.history(enc.repeat_interleave(sample, 0), actions.repeat_interleave(sample, 0))
                free = posts[0]

                def append(mode, t, prob, base, eventprob=None):
                    for j, ix in enumerate(ids):
                        ep = data.episodes[ix]
                        rr = ep['rows']
                        labels = ep['labels']
                        root = ep['root_id']
                        ri = max(0, t - 1)
                        lab = labels[ri]
                        controls = lab['control_samples_including_start'] - 1 if t else -1
                        stat = ('success' if lab['success'] else 'zero_control' if controls == 0 else 'executed_failure') if t else 'root_boundary'
                        target = labels[0]['before'] if t == 0 else labels[t - 1]['after']
                        before = labels[0]['before'] if base == 0 else labels[base - 1]['after']
                        prefix = [labels[k]['control_samples_including_start'] - 1 for k in range(t)]
                        row = {'root_id': root, 'episode_id': ep['episode_id'], 'split': data.split, 'role': rr[0]['dev_role'] if data.split == 'externalC' else data.split, 'mode': mode, 'step': t, 'prob': prob[j], 'cat': target['categorical'], 'cat_valid': target['categorical_valid'], 'binary': target['binary'], 'binary_valid': target['binary_valid'], 'base_cat': before['categorical'], 'base_valid': before['categorical_valid'], 'base_binary': before['binary'], 'base_binary_valid': before['binary_valid'], 'controls': controls, 'status': stat, 'skill': rr[ri]['skill'] if t else 'root', 'all_prefix_actuated': bool(t and all((v > 0 for v in prefix))), 'event_prob': eventprob[j] if eventprob is not None else np.full(4, np.nan), 'event_target': np.array([float(lab['success'])] + lab['events']) if t else np.full(4, np.nan), 'event_valid': np.array([True] + lab['events_valid']) if t else np.zeros(4, bool)}
                        records.append(row)
                for t in range(5):
                    append('current', t, probabilities(m, posts[t], n, sample), max(0, t - 1))
                    for j, ix in enumerate(ids):
                        ep = data.episodes[ix]
                        l = ep['labels'][0]['before'] if t == 0 else ep['labels'][t - 1]['after']
                        features.append({'x': ee[j, t], 'root_id': ep['root_id'], 'episode_id': ep['episode_id'], 'role': ep['rows'][0]['dev_role'] if data.split == 'externalC' else data.split, 'step': t, 'cat': l['categorical'], 'cat_valid': l['categorical_valid'], 'binary': l['binary'], 'binary_valid': l['binary_valid']})
                        qprobs.append(qq[j, t])
                        hstates.append(hs[j, t])
                for t in range(4):
                    wrong = torch.tensor([1, 4, 5, 0, 2, 3], device='cuda')[b['skill'][:, t]]
                    args = torch.tensor([static_arguments(data.skills[int(j)]) for j in wrong], device='cuda')
                    wa = m.action(wrong, args, b['budget'][:, t]).repeat_interleave(sample, 0)
                    ca = actions[:, t].repeat_interleave(sample, 0)
                    rng = torch.cuda.get_rng_state()
                    wp, _ = m.imagine(posts[t], wa)
                    torch.cuda.set_rng_state(rng)
                    cp, _ = m.imagine(posts[t], ca)
                    event = m.outcome_logits(posts[t], ca, cp).float().sigmoid().reshape(n, sample, 4).mean(1).cpu().numpy()
                    append('one_step', t + 1, probabilities(m, cp, n, sample), t, event)
                    append('wrong_action', t + 1, probabilities(m, wp, n, sample), t)
                    append('persistence', t + 1, probabilities(m, posts[t], n, sample), t)
                    free, _ = m.imagine(free, ca)
                    if t in [0, 1, 2, 3]:
                        append('free' + str(t + 1), t + 1, probabilities(m, free, n, sample), 0)
                        append('root_persistence' + str(t + 1), t + 1, probabilities(m, posts[0], n, sample), 0)
            if off % 40 == 0:
                print('EVALUATE', arm, seed, data.split, off + len(ids), '/', len(data.episodes), flush=True)
    prefix = f'{arm}_seed{seed}_{data.split}'
    savearrays(E / 'data' / f'predictions_{prefix}.npz', records)
    savearrays(E / 'data' / f'features_{prefix}.npz', features)
    e = np.stack([f['x'] for f in features]).astype(np.float64)
    q = np.stack(qprobs).astype(np.float64)
    h = np.stack(hstates).astype(np.float64)
    centered = e - e.mean(0)
    eig = np.maximum(np.linalg.eigvalsh(centered.T @ centered / max(1, len(e) - 1)), 0)
    ev = eig / max(eig.sum(), 1e-30)
    er = float(np.exp(-(ev * np.log(np.maximum(ev, 1e-30))).sum()))
    hc = -(q * np.log(np.maximum(q, 1e-30))).sum(-1).mean(0)
    qm = q.mean(0)
    hm = -(qm * np.log(np.maximum(qm, 1e-30))).sum(-1)
    roots = np.array([f['root_id'] for f in features])
    rootmean = np.stack([e[roots == r].mean(0) for r in sorted(set(roots))])
    rootstd = rootmean.std(0)
    one = [r for r in records if r['mode'] == 'one_step']
    wrong = [r for r in records if r['mode'] == 'wrong_action']
    assert all(((a['episode_id'], a['step']) == (b['episode_id'], b['step']) for a, b in zip(one, wrong)))
    tv = np.array([np.abs(a['prob'][:24].reshape(3, 8) - b['prob'][:24].reshape(3, 8)).sum(-1).mean() / 2 for a, b in zip(one, wrong)])
    result = {'utc': now(), 'arm': arm, 'seed': seed, 'split': data.split, 'roots': len(set(roots)), 'boundary_records': len(e), 'prediction_rows': len(records), 'seconds': time.perf_counter() - begin, 'E_std_mean': float(e.std(0).mean()), 'E_std_min': float(e.std(0).min()), 'E_lowstd_dimensions_below1e-4': int((e.std(0) < 0.0001).sum()), 'E_effective_rank': er, 'E_between_root_mean_std': float(rootstd.mean()), 'posterior_conditional_entropy_mean_nats': float(hc.mean()), 'posterior_marginal_entropy_mean_nats': float(hm.mean()), 'posterior_entropy_gap_mean_nats': float((hm - hc).mean()), 'posterior_argmax_categories_per_variable': [int(len(np.unique(q[:, j].argmax(-1)))) for j in range(32)], 'deterministic_h_std_mean': float(h.std(0).mean()), 'action_object_TV_mean': float(tv.mean()), 'support_hard_negative_FPR': None, 'support_denominator': 0, 'no_D': True, 'no_new_fitting': True}
    save(E / 'results' / f'collapse_{prefix}.json', result)
    return result
