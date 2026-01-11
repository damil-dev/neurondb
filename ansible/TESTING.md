# Ansible Testing and Validation

This document describes the testing and validation process for the Ansible infrastructure provisioning.

## Validation Script

Run the validation script to check for common issues:

```bash
cd ansible
./validate.sh
```

The script checks for:
- Required files and directories
- YAML syntax (if PyYAML is installed)
- Common Ansible issues
- Security concerns

## Manual Testing

### 1. Syntax Check (if Ansible is installed)

```bash
ansible-playbook --syntax-check playbooks/site.yml
ansible-playbook --syntax-check playbooks/infrastructure.yml
ansible-playbook --syntax-check playbooks/deploy-neurondb.yml
```

### 2. Dry Run (Check Mode)

```bash
# Test without making changes
ansible-playbook playbooks/site.yml --check

# Test specific playbook
ansible-playbook playbooks/infrastructure.yml --check --limit development
```

### 3. Inventory Validation

```bash
# Validate inventory
ansible-inventory -i inventory/hosts.yml --list

# Test connectivity
ansible all -i inventory/hosts.yml -m ping
```

## Known Issues Fixed

### 1. PostgreSQL Modules
- **Issue**: Used `postgresql_db`, `postgresql_user`, `postgresql_ext`, `postgresql_query` which require `community.postgresql` collection
- **Fix**: Replaced with shell commands using `psql` for better compatibility

### 2. Conditional Import Playbook
- **Issue**: `when` clause cannot be used with `import_playbook`
- **Fix**: Removed conditional imports, conditions handled inside playbooks

### 3. ansible_local Variable
- **Issue**: `ansible_local` doesn't exist in Ansible
- **Fix**: Replaced with `stat` module to check for script existence

### 4. playbook_dir Variable
- **Issue**: `playbook_dir` may not be available in all contexts
- **Fix**: Use `ansible_env.HOME` or check script existence first

### 5. Make Module Parameters
- **Issue**: `make` module doesn't support `params` parameter
- **Fix**: Replaced with `shell` module using proper environment variables

### 6. Extension File Verification
- **Issue**: Incorrect use of `failed_when` with loop results
- **Fix**: Separated verification into separate task with proper loop handling

### 7. Source Directory Paths
- **Issue**: Inconsistent source directory paths for components
- **Fix**: Standardized to use `/opt/neurondb-source` for all components

### 8. Missing Variables
- **Issue**: `postgresql_group` not defined
- **Fix**: Added to `roles/postgresql/vars/main.yml`

### 9. Build User Permissions
- **Issue**: Build tasks running as root
- **Fix**: Added `become_user: "{{ neurondb_user }}"` to build tasks

## Testing Checklist

- [ ] All playbooks pass syntax check
- [ ] All roles have required files (tasks/main.yml, vars/main.yml, handlers/main.yml)
- [ ] Inventory file is properly configured
- [ ] Group variables are set for all environments
- [ ] No hardcoded passwords (use vault)
- [ ] All password tasks have `no_log: true`
- [ ] Service files are properly templated
- [ ] Integration with existing scripts works

## Security Considerations

1. **Secrets Management**: Use Ansible Vault for all sensitive data
2. **Password Tasks**: All tasks using passwords should have `no_log: true`
3. **SSH Keys**: Use SSH key-based authentication
4. **Firewall**: Enable firewall rules in production
5. **Service Users**: Use dedicated service users, not root

## Troubleshooting

### Common Errors

1. **"Module not found"**: Install required Ansible collections
   ```bash
   ansible-galaxy collection install -r requirements.yml
   ```

2. **"Permission denied"**: Check SSH access and sudo permissions

3. **"Variable not defined"**: Check group_vars and host_vars files

4. **"Service failed to start"**: Check logs with `journalctl -u service-name`

## Continuous Improvement

- Add integration tests with testinfra
- Add molecule tests for roles
- Add CI/CD validation
- Expand validation script checks

