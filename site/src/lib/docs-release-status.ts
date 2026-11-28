// Documentation follows main; a published executable keeps its frozen feature
// set. New commands must be marked Development until a release includes them.
export const STABLE_RELEASE = '0.1.3';
export const STABLE_RELEASE_URL = `https://github.com/zolotukhin/zinc/releases/tag/v${STABLE_RELEASE}`;
export const STABLE_VALIDATION_PATH = '/zinc/validation/release-candidate-0.1.3.json';
export const CLIENT_VALIDATION_RELEASE = '0.1.2';
export const CLIENT_VALIDATION_PATH = '/zinc/validation/release-candidate-0.1.2-2026-11-27.json';

export function docsReleaseStatus(slug: string) {
  if (slug === 'getting-started')
    return {
      channel: 'stable',
      label: `Install stable Linux ${STABLE_RELEASE}`,
      description: 'Preview and source-build options are listed separately.',
    };
  if (['openclaw', 'opencode'].includes(slug))
    return {
      channel: 'stable',
      label: `Client setup for ZINC ${STABLE_RELEASE}`,
      description:
        'Use the pinned client recipe; model-specific evidence is linked under Tested scope.',
    };
  if (['running-zinc', 'api', 'hardware-requirements', 'rocm'].includes(slug)) {
    return {
      channel: 'mixed',
      label: `Stable ${STABLE_RELEASE} + development`,
      description: 'Development features and source-specific reports are marked separately.',
    };
  }
  return {
    channel: 'development',
    label: 'Source reference and engineering evidence',
    description: 'Use the version and hardware recorded in each report when applying its results.',
  };
}
