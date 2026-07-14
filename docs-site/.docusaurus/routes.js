import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/hacs-llm-sandbox/',
    component: ComponentCreator('/hacs-llm-sandbox/', 'cc5'),
    routes: [
      {
        path: '/hacs-llm-sandbox/',
        component: ComponentCreator('/hacs-llm-sandbox/', 'bb5'),
        routes: [
          {
            path: '/hacs-llm-sandbox/',
            component: ComponentCreator('/hacs-llm-sandbox/', '8ec'),
            routes: [
              {
                path: '/hacs-llm-sandbox/architecture/facade-surface',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/facade-surface', '338'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/guidance-and-recovery',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/guidance-and-recovery', 'b10'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/integration-lifecycle',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/integration-lifecycle', '650'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/llm-api-registration',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/llm-api-registration', '91f'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/monty-execution',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/monty-execution', 'fe3'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/overview',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/overview', '574'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/recorder-tools',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/recorder-tools', '401'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/service-call-gating',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/service-call-gating', '4ae'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/snapshot-pipeline',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/snapshot-pipeline', 'ada'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/vision-tool',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/vision-tool', 'd61'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/action-safety',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/action-safety', 'f9b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/assist-tool-calling',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/assist-tool-calling', '51c'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/model-quality-and-cost',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/model-quality-and-cost', 'e58'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/recorder-and-sql',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/recorder-and-sql', 'ad7'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/sandbox-boundaries',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/sandbox-boundaries', '7ff'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/snapshots-and-visibility',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/snapshots-and-visibility', '235'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/configuration/actions',
                component: ComponentCreator('/hacs-llm-sandbox/configuration/actions', 'eff'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/configuration/execution-limits',
                component: ComponentCreator('/hacs-llm-sandbox/configuration/execution-limits', 'a39'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/configuration/overview',
                component: ComponentCreator('/hacs-llm-sandbox/configuration/overview', '1a3'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/configuration/prompt-profiles',
                component: ComponentCreator('/hacs-llm-sandbox/configuration/prompt-profiles', '289'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/configuration/visibility',
                component: ComponentCreator('/hacs-llm-sandbox/configuration/visibility', '2f8'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/development/contributing',
                component: ComponentCreator('/hacs-llm-sandbox/development/contributing', '9d3'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/development/eval-harness',
                component: ComponentCreator('/hacs-llm-sandbox/development/eval-harness', '130'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/development/local-setup',
                component: ComponentCreator('/hacs-llm-sandbox/development/local-setup', '5a6'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/development/validation',
                component: ComponentCreator('/hacs-llm-sandbox/development/validation', 'd2a'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/installation/enable-in-assist',
                component: ComponentCreator('/hacs-llm-sandbox/installation/enable-in-assist', '0a8'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/installation/install-with-hacs',
                component: ComponentCreator('/hacs-llm-sandbox/installation/install-with-hacs', '1fe'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/installation/prerequisites',
                component: ComponentCreator('/hacs-llm-sandbox/installation/prerequisites', '81a'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/operations/choosing-a-model',
                component: ComponentCreator('/hacs-llm-sandbox/operations/choosing-a-model', '989'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/operations/performance-and-token-cost',
                component: ComponentCreator('/hacs-llm-sandbox/operations/performance-and-token-cost', '258'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/operations/privacy-and-security',
                component: ComponentCreator('/hacs-llm-sandbox/operations/privacy-and-security', '3ab'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/configuration-options',
                component: ComponentCreator('/hacs-llm-sandbox/reference/configuration-options', 'fba'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/error-keys',
                component: ComponentCreator('/hacs-llm-sandbox/reference/error-keys', 'b33'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/limits',
                component: ComponentCreator('/hacs-llm-sandbox/reference/limits', 'b9b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/monty-globals',
                component: ComponentCreator('/hacs-llm-sandbox/reference/monty-globals', 'f8f'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/sql-schema',
                component: ComponentCreator('/hacs-llm-sandbox/reference/sql-schema', 'ebb'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/tools/execute-home-code',
                component: ComponentCreator('/hacs-llm-sandbox/reference/tools/execute-home-code', 'd9c'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/tools/get-automation',
                component: ComponentCreator('/hacs-llm-sandbox/reference/tools/get-automation', '4ef'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/tools/get-camera-image',
                component: ComponentCreator('/hacs-llm-sandbox/reference/tools/get-camera-image', '3bd'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/tools/get-history',
                component: ComponentCreator('/hacs-llm-sandbox/reference/tools/get-history', '520'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/tools/get-logbook',
                component: ComponentCreator('/hacs-llm-sandbox/reference/tools/get-logbook', '73b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/tools/get-statistics',
                component: ComponentCreator('/hacs-llm-sandbox/reference/tools/get-statistics', 'e03'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/release-notes',
                component: ComponentCreator('/hacs-llm-sandbox/release-notes', 'cad'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/troubleshooting',
                component: ComponentCreator('/hacs-llm-sandbox/troubleshooting', '925'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/upgrades',
                component: ComponentCreator('/hacs-llm-sandbox/upgrades', '62c'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/camera-images',
                component: ComponentCreator('/hacs-llm-sandbox/usage/camera-images', '976'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/example-prompts',
                component: ComponentCreator('/hacs-llm-sandbox/usage/example-prompts', '4cf'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/history-statistics-logbook',
                component: ComponentCreator('/hacs-llm-sandbox/usage/history-statistics-logbook', 'cd9'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/quickstart',
                component: ComponentCreator('/hacs-llm-sandbox/usage/quickstart', 'e0d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/service-actions',
                component: ComponentCreator('/hacs-llm-sandbox/usage/service-actions', '9e3'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/state-and-registry-questions',
                component: ComponentCreator('/hacs-llm-sandbox/usage/state-and-registry-questions', 'b86'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/',
                component: ComponentCreator('/hacs-llm-sandbox/', 'ee8'),
                exact: true,
                sidebar: "docsSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
