export function initBenchmarks() {
  const root = document.querySelector<HTMLElement>('[data-benchmarks]');
  if (!root || root.dataset.initialized) return;
  root.dataset.initialized = 'true';
  const tabs = Array.from(root.querySelectorAll<HTMLButtonElement>('[data-bench-target-tab]'));
  const panels = Array.from(root.querySelectorAll<HTMLElement>('[data-bench-target-panel]'));
  const status = root.querySelector<HTMLElement>('[data-share-status]');
  let state = { target: tabs[0]?.dataset.benchTargetTab ?? '', workload: 'core', mtp: 'off' };

  function render() {
    for (const tab of tabs) {
      const active = tab.dataset.benchTargetTab === state.target;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    }
    for (const panel of panels) {
      panel.hidden = panel.dataset.benchTargetPanel !== state.target;
      for (const button of panel.querySelectorAll<HTMLButtonElement>('[data-bench-scenario]')) {
        button.setAttribute(
          'aria-pressed',
          String(button.dataset.benchScenario === state.workload),
        );
      }
      for (const button of panel.querySelectorAll<HTMLButtonElement>('[data-mtp-mode]')) {
        button.setAttribute('aria-pressed', String(button.dataset.mtpMode === state.mtp));
      }
      for (const mode of panel.querySelectorAll<HTMLElement>('[data-bench-mode-panel]')) {
        mode.hidden = mode.dataset.benchModePanel !== state.mtp;
        for (const scenario of mode.querySelectorAll<HTMLElement>('[data-bench-scenario-panel]')) {
          scenario.hidden = scenario.dataset.benchScenarioPanel !== state.workload;
        }
      }
    }
  }

  function normalize() {
    const panel =
      panels.find((candidate) => candidate.dataset.benchTargetPanel === state.target) ?? panels[0];
    if (!panel) return;
    state.target = panel.dataset.benchTargetPanel!;
    if (
      !Array.from(panel.querySelectorAll<HTMLElement>('[data-bench-scenario]')).some(
        (button) => button.dataset.benchScenario === state.workload,
      )
    )
      state.workload = panel.dataset.defaultScenario ?? 'core';
    if (state.mtp !== 'on' || !panel.querySelector('[data-mtp-mode="on"]')) state.mtp = 'off';
  }

  function updateUrl(push = false) {
    const url = new URL(location.href);
    url.hash = state.target;
    if (state.workload === 'core') url.searchParams.delete('workload');
    else url.searchParams.set('workload', state.workload);
    if (state.mtp === 'off') url.searchParams.delete('mtp');
    else url.searchParams.set('mtp', 'on');
    if (url.href !== location.href) history[push ? 'pushState' : 'replaceState'](null, '', url);
  }

  function readUrl() {
    const url = new URL(location.href);
    state = {
      target: url.hash.slice(1) || tabs[0]?.dataset.benchTargetTab || '',
      workload: url.searchParams.get('workload') ?? 'core',
      mtp: url.searchParams.get('mtp') === 'on' ? 'on' : 'off',
    };
    normalize();
    render();
  }

  function select(next: Partial<typeof state>) {
    state = { ...state, ...next };
    normalize();
    render();
    updateUrl(true);
    if (status) status.textContent = '';
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select({ target: tab.dataset.benchTargetTab! }));
    tab.addEventListener('keydown', (event) => {
      let next: number;
      switch (event.key) {
        case 'ArrowRight':
        case 'ArrowDown':
          next = (index + 1) % tabs.length;
          break;
        case 'ArrowLeft':
        case 'ArrowUp':
          next = (index + tabs.length - 1) % tabs.length;
          break;
        case 'Home':
          next = 0;
          break;
        case 'End':
          next = tabs.length - 1;
          break;
        default:
          return;
      }
      event.preventDefault();
      tabs[next].focus();
      select({ target: tabs[next].dataset.benchTargetTab! });
    });
  });

  for (const panel of panels) {
    for (const button of panel.querySelectorAll<HTMLButtonElement>('[data-bench-scenario]')) {
      button.addEventListener('click', () => select({ workload: button.dataset.benchScenario! }));
    }
    for (const button of panel.querySelectorAll<HTMLButtonElement>('[data-mtp-mode]')) {
      button.addEventListener('click', () => select({ mtp: button.dataset.mtpMode! }));
    }
    panel.querySelector('[data-bench-share]')?.addEventListener('click', async (event) => {
      updateUrl();
      const button = event.currentTarget as HTMLButtonElement;
      try {
        await navigator.clipboard.writeText(location.href);
        button.textContent = 'Link copied';
        if (status)
          status.textContent = 'Link copied. It includes the platform, workload, and MTP setting.';
      } catch {
        if (status) status.textContent = 'Copy the URL from your address bar to share this view.';
      }
      window.setTimeout(() => {
        button.textContent = 'Copy link to this view';
      }, 2000);
    });
  }

  window.addEventListener('popstate', readUrl);
  window.addEventListener('hashchange', readUrl);
  readUrl();
}
