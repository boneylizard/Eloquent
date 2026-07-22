import ProfileSelector from '../components/ProfileSelector';
import SimpleUserProfileEditor from '../components/SimpleUserProfileEditor';
import { useMemory } from '../contexts/MemoryContext';

const UserProfilesPage = () => {
  const { profiles, isLoading } = useMemory();
  const hasProfiles = Array.isArray(profiles) && profiles.length > 0;

  return (
    <div className="w-full min-h-screen p-2 md:p-4">
      <div className="mx-auto max-w-6xl space-y-4">
        <h2 className="text-2xl font-bold">User Profiles</h2>
        <div className="rounded-lg border bg-card p-4 md:p-6">
          <ProfileSelector />
          {!isLoading && hasProfiles && (
            <>
              <div className="my-6 border-t border-border/60" />
              <SimpleUserProfileEditor />
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default UserProfilesPage;
