                  ┌────────────────────────────────────────┐
                  │       Procedural Scene Generator       │
                  │   (Generates 2D shape canvas image)    │
                  └───────────────────┬────────────────────┘
                                      │
                                [ Raw Image ]
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │            NEURAL MODULE               │
                  │       Object Detection (YOLOv8)        │
                  │  Outputs: [bbox, class, color, x, y]   │
                  └───────────────────┬────────────────────┘
                                      │
                              [ Scene Graph ]
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                             SYMBOLIC MODULE                              │
│                                                                          │
│   ┌───────────────────────────┐        ┌──────────────────────────────┐  │
│   │     Domain Specific       │        │        Rule Engine           │  │
│   │       Language (DSL)      ├───────►│  Evaluates candidate rules   │  │
│   │ (exists, count, spatial)  │        │   against the Scene Graph    │  │
│   └───────────────────────────┘        └──────────────┬───────────────┘  │
│                                                       │                  │
│                                              [ Valid Rules ]             │
│                                                       │                  │
│                                                       ▼                  │
│                                        ┌──────────────────────────────┐  │
│                                        │    Active Learning Engine    │  │
│                                        │  Selects candidate scene that│  │
│                                        │  splits hypothesis space 50% │  │
│                                        └──────────────┬───────────────┘  │
└───────────────────────────────────────────────────────┼──────────────────┘
                                                        │
                                                [ Ask User: Y/N? ]
                                                        │
                                                        ▼
                                          ┌───────────────────────────┐
                                          │ Human User / Teacher Feed │
                                          └───────────────────────────┘