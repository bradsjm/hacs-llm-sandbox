---
title: Example Prompts
description: Practical prompts that match the tool set.
---

# Example Prompts

## Current home state

```text
Which lights are on right now, grouped by floor and area?
```

```text
Find sensors with unavailable or unknown state and summarize them by integration.
```

## Diagnostics

```text
List visible battery sensors below 25 percent and include the device area.
```

```text
Are there any open repairs or persistent notifications I should handle?
```

## History and statistics

```text
Did the front door open after 10pm last night? Include nearby logbook activity.
```

```text
Which room had the highest average humidity over the last 24 hours?
```

## Composed recorder reasoning

```text
If the living-room light was turned on after midnight and is still on, turn it off.
```

This should use one `execute_home_code` call: inspect the frozen state, `await hass.logbook(...)`, and conditionally act without a sequential recorder round.

## Independent direct reads

```text
Show the front-door activity and the bedroom humidity average for the last day.
```

These independent requests can use `get_logbook` and `get_statistics` in parallel.

## Visual checks

```text
Check the front porch camera and tell me if there is a package.
```

## Actions, only if enabled

```text
Turn off visible lights that have been on for more than two hours.
```

Keep action prompts explicit. Avoid asking the model to make broad changes until you have watched how your selected model behaves.
