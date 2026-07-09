import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/hacs-llm-sandbox/',
    component: ComponentCreator('/hacs-llm-sandbox/', '852'),
    routes: [
      {
        path: '/hacs-llm-sandbox/',
        component: ComponentCreator('/hacs-llm-sandbox/', 'ae7'),
        routes: [
          {
            path: '/hacs-llm-sandbox/',
            component: ComponentCreator('/hacs-llm-sandbox/', '004'),
            routes: [
              {
                path: '/hacs-llm-sandbox/architecture/facade-surface',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/facade-surface', '40b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/guidance-and-recovery',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/guidance-and-recovery', 'c69'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/integration-lifecycle',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/integration-lifecycle', '1ac'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/llm-api-registration',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/llm-api-registration', '4e9'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/monty-execution',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/monty-execution', 'a08'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/overview',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/overview', '32e'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/recorder-tools',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/recorder-tools', '84d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/service-call-gating',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/service-call-gating', 'd93'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/snapshot-pipeline',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/snapshot-pipeline', '66d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/architecture/vision-tool',
                component: ComponentCreator('/hacs-llm-sandbox/architecture/vision-tool', 'ff8'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/action-safety',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/action-safety', 'dc4'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/assist-tool-calling',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/assist-tool-calling', 'ee4'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/model-quality-and-cost',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/model-quality-and-cost', 'c2f'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/recorder-and-sql',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/recorder-and-sql', 'f86'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/sandbox-boundaries',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/sandbox-boundaries', '306'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/concepts/snapshots-and-visibility',
                component: ComponentCreator('/hacs-llm-sandbox/concepts/snapshots-and-visibility', '4fd'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/configuration/actions',
                component: ComponentCreator('/hacs-llm-sandbox/configuration/actions', 'cb2'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/configuration/execution-limits',
                component: ComponentCreator('/hacs-llm-sandbox/configuration/execution-limits', '5d4'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/configuration/overview',
                component: ComponentCreator('/hacs-llm-sandbox/configuration/overview', '9d8'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/configuration/prompt-profiles',
                component: ComponentCreator('/hacs-llm-sandbox/configuration/prompt-profiles', '97f'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/configuration/visibility',
                component: ComponentCreator('/hacs-llm-sandbox/configuration/visibility', '881'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/development/contributing',
                component: ComponentCreator('/hacs-llm-sandbox/development/contributing', 'ccc'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/development/eval-harness',
                component: ComponentCreator('/hacs-llm-sandbox/development/eval-harness', '7ef'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/development/local-setup',
                component: ComponentCreator('/hacs-llm-sandbox/development/local-setup', 'c2b'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/development/validation',
                component: ComponentCreator('/hacs-llm-sandbox/development/validation', 'b06'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/installation/enable-in-assist',
                component: ComponentCreator('/hacs-llm-sandbox/installation/enable-in-assist', '0fe'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/installation/install-with-hacs',
                component: ComponentCreator('/hacs-llm-sandbox/installation/install-with-hacs', '1ae'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/installation/prerequisites',
                component: ComponentCreator('/hacs-llm-sandbox/installation/prerequisites', '2a3'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/operations/choosing-a-model',
                component: ComponentCreator('/hacs-llm-sandbox/operations/choosing-a-model', 'de0'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/operations/performance-and-token-cost',
                component: ComponentCreator('/hacs-llm-sandbox/operations/performance-and-token-cost', 'b72'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/operations/privacy-and-security',
                component: ComponentCreator('/hacs-llm-sandbox/operations/privacy-and-security', 'bb2'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/configuration-options',
                component: ComponentCreator('/hacs-llm-sandbox/reference/configuration-options', 'e00'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/error-keys',
                component: ComponentCreator('/hacs-llm-sandbox/reference/error-keys', 'f49'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/limits',
                component: ComponentCreator('/hacs-llm-sandbox/reference/limits', 'e98'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/monty-globals',
                component: ComponentCreator('/hacs-llm-sandbox/reference/monty-globals', '7e8'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/sql-schema',
                component: ComponentCreator('/hacs-llm-sandbox/reference/sql-schema', '495'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/tools/execute-home-code',
                component: ComponentCreator('/hacs-llm-sandbox/reference/tools/execute-home-code', '4e8'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/tools/get-camera-image',
                component: ComponentCreator('/hacs-llm-sandbox/reference/tools/get-camera-image', '2eb'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/tools/get-history',
                component: ComponentCreator('/hacs-llm-sandbox/reference/tools/get-history', '740'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/tools/get-logbook',
                component: ComponentCreator('/hacs-llm-sandbox/reference/tools/get-logbook', 'e5c'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/reference/tools/get-statistics',
                component: ComponentCreator('/hacs-llm-sandbox/reference/tools/get-statistics', '9aa'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/release-notes',
                component: ComponentCreator('/hacs-llm-sandbox/release-notes', 'b57'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/troubleshooting',
                component: ComponentCreator('/hacs-llm-sandbox/troubleshooting', '653'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/upgrades',
                component: ComponentCreator('/hacs-llm-sandbox/upgrades', 'c4d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/camera-images',
                component: ComponentCreator('/hacs-llm-sandbox/usage/camera-images', '03d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/example-prompts',
                component: ComponentCreator('/hacs-llm-sandbox/usage/example-prompts', '941'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/history-statistics-logbook',
                component: ComponentCreator('/hacs-llm-sandbox/usage/history-statistics-logbook', 'b0e'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/quickstart',
                component: ComponentCreator('/hacs-llm-sandbox/usage/quickstart', '8b7'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/service-actions',
                component: ComponentCreator('/hacs-llm-sandbox/usage/service-actions', '744'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/usage/state-and-registry-questions',
                component: ComponentCreator('/hacs-llm-sandbox/usage/state-and-registry-questions', 'f5d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/hacs-llm-sandbox/',
                component: ComponentCreator('/hacs-llm-sandbox/', '243'),
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
