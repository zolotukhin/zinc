// Documentation follows main; a published executable keeps its frozen feature
// set. New commands must be marked Development until a release includes them.
export const STABLE_RELEASE = '0.1.0';
export const STABLE_RELEASE_URL = `https://github.com/zolotukhin/zinc/releases/tag/v${STABLE_RELEASE}`;

export function docsReleaseStatus(slug: string) {
  if (slug === 'binary-releases')
    return { channel: 'stable', label: `Stable Linux ${STABLE_RELEASE}` };
  if (
    [
      'getting-started',
      'running-zinc',
      'api',
      'agent-setup',
      'hardware-requirements',
      'rocm',
      'openclaw',
      'opencode',
    ].includes(slug)
  ) {
    return {
      channel: 'mixed',
      label: `Stable ${STABLE_RELEASE} + development`,
    };
  }
  return { channel: 'development', label: 'Current source reference' };
}
