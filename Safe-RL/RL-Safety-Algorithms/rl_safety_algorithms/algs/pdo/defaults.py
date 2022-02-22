def defaults():
    return dict(
        actor='mlp',
        ac_kwargs={
            'pi': {'hidden_sizes': (64, 64),
                   'activation': 'tanh'},
            'val': {'hidden_sizes': (64, 64),
                    'activation': 'tanh'}
        },
        adv_estimation_method='gae',
        epochs=300,  # 9.8M steps
        gamma=0.99,
        lambda_lr=0.001,
        lambda_optimizer='Adam',
        steps_per_epoch=64 * 1000,
        target_kl=0.001,
        use_exploration_noise_anneal=True
    )
