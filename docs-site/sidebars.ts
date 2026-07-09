import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Installation',
      items: [
        'installation/prerequisites',
        'installation/install-with-hacs',
        'installation/enable-in-assist',
      ],
    },
    {
      type: 'category',
      label: 'Configuration',
      items: [
        'configuration/overview',
        'configuration/visibility',
        'configuration/actions',
        'configuration/execution-limits',
        'configuration/prompt-profiles',
      ],
    },
    {
      type: 'category',
      label: 'Using Assist Agent Sandbox',
      items: [
        'usage/quickstart',
        'usage/state-and-registry-questions',
        'usage/history-statistics-logbook',
        'usage/camera-images',
        'usage/service-actions',
        'usage/example-prompts',
      ],
    },
    {
      type: 'category',
      label: 'Concepts',
      items: [
        'concepts/assist-tool-calling',
        'concepts/snapshots-and-visibility',
        'concepts/sandbox-boundaries',
        'concepts/recorder-and-sql',
        'concepts/action-safety',
        'concepts/model-quality-and-cost',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/overview',
        'architecture/integration-lifecycle',
        'architecture/llm-api-registration',
        'architecture/snapshot-pipeline',
        'architecture/monty-execution',
        'architecture/facade-surface',
        'architecture/recorder-tools',
        'architecture/vision-tool',
        'architecture/service-call-gating',
        'architecture/guidance-and-recovery',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/configuration-options',
        {
          type: 'category',
          label: 'Tools',
          items: [
            'reference/tools/execute-home-code',
            'reference/tools/get-history',
            'reference/tools/get-statistics',
            'reference/tools/get-logbook',
            'reference/tools/get-camera-image',
          ],
        },
        'reference/monty-globals',
        'reference/sql-schema',
        'reference/limits',
        'reference/error-keys',
      ],
    },
    {
      type: 'category',
      label: 'Operations',
      items: [
        'operations/privacy-and-security',
        'operations/performance-and-token-cost',
        'operations/choosing-a-model',
        'troubleshooting',
        'upgrades',
      ],
    },
    {
      type: 'category',
      label: 'Development',
      items: [
        'development/local-setup',
        'development/validation',
        'development/eval-harness',
        'development/contributing',
      ],
    },
    'release-notes',
  ],
};

export default sidebars;
