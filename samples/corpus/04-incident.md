# Field incident response (IR-4)

When a K-Walk collides, e-stops, or tips a rack at a customer site, follow this process. Do not apologize in the customer chat first and backfill the process later.

## Severity

- **SEV-1**: injury, or equipment cannot resume work within 2 hours
- **SEV-2**: line down more than 30 minutes, no injury
- **SEV-3**: single-unit failure with a hot spare or a workaround

## SEV-1 clock

1. Minute 0: hit e-stop on site, evacuate people, **call the duty manager** (do not Slack first)
2. Minute 15: open a bridge in the `incidents` channel, sync the customer single point of contact
3. Minute 60: written status (what happened, risk, next step, next update time)
4. 24 hours: internal retro draft; formal RCA within **5 working days**

## Forbidden

- Do not upload customer camera frames to a public LLM “to see how it crashed”
- Do not promise a software root cause to the customer before the RCA is done
