for seed in [0, 1]:
    cp = torch.load(CELL / f'models/joint_seed{seed}/latest.pt', map_location='cpu', weights_only=False)
    assert cp['update'] == 6000
    schedule = schedules[seed]
    init_hash = None
    for arm in ARMS:
        m = ObservationStateWorldModel(skill_count=6, event_count=3).cuda()
        load_shared(m, cp['model'])
        before = statehash(m)
        init_hash = init_hash or before
        assert before == init_hash
        opt = torch.optim.AdamW(shared_parameters(m), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)
        opt.load_state_dict(copy.deepcopy(cp['optimizer']))
        add_new_optimizer_group(opt, m)
        for g in opt.param_groups:
            g['lr'] = 0.0001
        m.train()
        begin = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        curve = []
        candidates = []
        log = (S / f'logs/train_{arm}_seed{seed}.jsonl').open('w')
        for step in range(1, 1001):
            opt.zero_grad(set_to_none=True)
            values = {}
            for acc in range(2):
                examples = [oldcache[int(i)] for i in schedule['old'][step - 1, acc]] + [newcache[arm][int(i)] for i in schedule['new'][step - 1, acc]]
                b = {k: torch.stack([x[k] for x in examples]).cuda() for k in examples[0]}
                torch.manual_seed(int(schedule['latent'][step - 1, acc]))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    total, current, future, terms, enc, terminal = losses(m, b, 3, w)
                objective = total + terminal
                assert torch.isfinite(objective)
                backward_routed(m, objective, current + terminal, enc, 'joint', 2)
                for k, v in dict(terms, terminal=terminal, objective=objective).items():
                    values[k] = values.get(k, 0) + float(v.detach()) / 2
                del b, total, current, future, terms, enc, terminal, objective
            en, rn = clip_groups(m)
            opt.step()
            if step == 1 or step % 50 == 0:
                row = {'utc': now(), 'seed': seed, 'arm': arm, 'step': step, 'seconds': time.monotonic() - begin, 'peak_GiB': torch.cuda.max_memory_allocated() / 1024 ** 3, 'E_norm': en, 'R_norm': rn, **values}
                curve.append(row)
                log.write(json.dumps(row) + '\n')
                log.flush()
                save(S / 'results/training_progress.json', row)
                print('FIT', json.dumps(row), flush=True)
            if step in [500, 1000]:
                m.eval()
                devscore = score(m, vcache, w)
                path = S / f'models/{arm}_seed{seed}_step{step}.pt'
                torch.save({'model': m.state_dict(), 'seed': seed, 'arm': arm, 'continuation_update': step, 'initial_hash': before, 'protocol_sha256': sha(S / 'configs/protocol.json')}, path)
                candidates.append({'update': step, 'development_objective': devscore, 'path': str(path), 'sha256': sha(path)})
                print('DEV', seed, arm, step, devscore, flush=True)
                m.train()
        chosen = min(candidates, key=lambda x: (x['development_objective'], x['update']))
        receipt = {'utc': now(), 'seed': seed, 'arm': arm, 'updates': 1000, 'initial_hash': before, 'final_hash': statehash(m), 'seconds': time.monotonic() - begin, 'peak_GiB': torch.cuda.max_memory_allocated() / 1024 ** 3, 'candidates': candidates, 'selected': chosen, 'curve': curve, 'old_samples': 8000, 'new_samples': 8000, 'same_terminal_loss_both_arms': True}
        save(S / f'results/train_{arm}_seed{seed}.json', receipt)
        results.append(receipt)
        selections.append({'seed': seed, 'arm': arm, **chosen})
        log.close()
        del m, opt
        torch.cuda.empty_cache()
    del cp
