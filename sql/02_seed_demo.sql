SET search_path TO movie_night, public;

INSERT INTO users (email, display_name, is_demo) VALUES
    ('ava@example.com',  'Ava',  true),
    ('ben@example.com',  'Ben',  true),
    ('cleo@example.com', 'Cleo', true)
ON CONFLICT (email) DO NOTHING;

INSERT INTO groups (name, created_by)
SELECT 'Friday Movie Night', (SELECT user_id FROM users WHERE email='ava@example.com')
WHERE NOT EXISTS (SELECT 1 FROM groups WHERE name = 'Friday Movie Night');

INSERT INTO group_members (group_id, user_id)
SELECT g.group_id, u.user_id
FROM groups g CROSS JOIN users u
WHERE g.name = 'Friday Movie Night' AND u.is_demo
ON CONFLICT DO NOTHING;
